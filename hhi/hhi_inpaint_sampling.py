"""
hhi_inpaint_sampling.py
------------------------
Prefix masked completion 和 overlap inpainting 推理逻辑。

训练和推理必须共用同一套"硬前缀观测 + 仅后缀更新"的 prefix 机制。

核心函数：
  apply_prefix_mask()       -- 把 prefix 区域 GT 写回 noisy_motion（训练 & 推理共用）
  merge_overlap_region()    -- phase 拼接时对 overlap 区间做 SLERP/线性融合
  sample_with_inpainting()  -- 推理时的完整采样循环（Flow Matching ODE）
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .hhi_constants import OVERLAP_N


# ── apply_prefix_mask ─────────────────────────────────────────────────────────

def apply_prefix_mask(
    noisy_motion : torch.Tensor,         # [B, W, D]  可修改
    gt_motion    : torch.Tensor,         # [B, W, D]  GT / 观测值
    prefix_mask  : torch.BoolTensor,     # [B, W]
) -> torch.Tensor:
    """
    强制把 prefix_mask=True 的区域用 GT/观测值覆盖 noisy_motion。

    合约（见 Section 6.3）：
      1. prefix 区域已经是只读观测值，不再额外加噪
      2. 后续 forward() 中 denoising 只能更新 ~prefix_mask 区域
      3. loss 侧通过 valid_mask = seq_mask & (~prefix_mask) 屏蔽 prefix

    返回更新后的 noisy_motion（in-place 修改并返回，caller 可忽略返回值）。
    """
    mask_expanded = prefix_mask.unsqueeze(-1).expand_as(noisy_motion)
    noisy_motion  = torch.where(mask_expanded, gt_motion, noisy_motion)
    return noisy_motion


# ── SLERP for quaternions ─────────────────────────────────────────────────────

def _slerp_quat(
    q0   : torch.Tensor,   # [..., 4]  xyzw
    q1   : torch.Tensor,   # [..., 4]
    t    : float,
) -> torch.Tensor:
    """球面线性插值（batch-friendly）。"""
    q0 = F.normalize(q0, dim=-1)
    q1 = F.normalize(q1, dim=-1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)   # cosine
    # 当 dot < 0 时翻转 q1 以取短弧
    q1  = torch.where(dot < 0, -q1, q1)
    dot = dot.abs()
    theta = torch.acos(dot.clamp(max=1.0 - 1e-6))
    sin_t = torch.sin(theta)
    # 退化（theta ≈ 0）时线性插值
    safe  = sin_t.abs() > 1e-6
    w0    = torch.where(safe, torch.sin((1 - t) * theta) / sin_t, torch.full_like(dot, 1 - t))
    w1    = torch.where(safe, torch.sin(t       * theta) / sin_t, torch.full_like(dot, t))
    return F.normalize(w0 * q0 + w1 * q1, dim=-1)


# ── merge_overlap_region ──────────────────────────────────────────────────────

def merge_overlap_region(
    old_phase   : torch.Tensor,       # [W_old, D_motion]  已落盘的旧段末尾
    new_phase   : torch.Tensor,       # [W_new, D_motion]  新生成段
    overlap_n   : int = OVERLAP_N,
    root_pos_slice : Tuple[int, int] = (0, 3),    # root translation channel slice
    root_rot_slice : Tuple[int, int] = (3, 9),    # root 6D rotation channel slice
) -> torch.Tensor:
    """
    对 overlap 区间内的 k-th 帧（k=0..overlap_n-1）做线性融合。

    融合规则（见 Section 13.10）：
      α_k = (overlap_n - k) / overlap_n   旧段权重（从 1.0 降到 0.1）
      β_k = k / overlap_n                  新段权重（从 0.0 升到 0.9）

    所有 channel（包括 root translation、root 6D rotation、body joints）均用线性插值。
    6D rotation 的正确融合方式是线性插值（不是 SLERP）：插值后的 6D 向量
    经 Gram-Schmidt 正交化即可还原为合法旋转矩阵，无需特殊处理。

    旧段末尾 overlap_n 帧 vs 新段首部 overlap_n 帧。
    返回融合后的 new_phase（首部 overlap_n 帧已替换）。
    """
    assert new_phase.shape[0] >= overlap_n, (
        f"new_phase length {new_phase.shape[0]} < overlap_n {overlap_n}"
    )
    result   = new_phase.clone()
    old_tail = old_phase[-overlap_n:]   # [overlap_n, D]
    new_head = new_phase[:overlap_n]    # [overlap_n, D]

    # 向量化线性插值：alpha/beta 从 [overlap_n] 广播到 [overlap_n, D]
    ks    = torch.arange(overlap_n, device=old_phase.device, dtype=old_phase.dtype)
    alpha = (overlap_n - ks) / overlap_n   # [overlap_n]  1.0 → 0.1
    beta  = ks / overlap_n                 # [overlap_n]  0.0 → 0.9
    merged = alpha[:, None] * old_tail + beta[:, None] * new_head   # [overlap_n, D]
    result[:overlap_n] = merged

    return result


def merge_overlap_region_numpy(
    old_phase   : np.ndarray,   # [W_old, D]
    new_phase   : np.ndarray,   # [W_new, D]
    overlap_n   : int = OVERLAP_N,
) -> np.ndarray:
    """numpy 版本，用于推理时在 CPU 上快速合并。"""
    old_t  = torch.from_numpy(old_phase)
    new_t  = torch.from_numpy(new_phase)
    merged = merge_overlap_region(old_t, new_t, overlap_n)
    return merged.numpy()


# ── Flow Matching sampling ────────────────────────────────────────────────────

def sample_with_inpainting(
    model           ,                        # HHIExecutorModel
    batch           : dict,
    n_steps         : int = 20,
    method          : str = "euler",         # "euler" | "rk4"
    device          : str = "cpu",
) -> torch.Tensor:
    """
    推理时的 Flow Matching ODE 采样。

    合约：
      - prefix_mask 区域在每步 ODE 迭代中保持 GT 观测值不变（apply_prefix_mask）
      - 只更新 ~prefix_mask 区域
      - 最终返回完整 [B, W, D] 序列（prefix + 生成后缀）

    Flow Matching ODE：
      dx/dt = v_theta(x_t, t, cond)
      x_0 ~ N(0, I)
      x_1 = 目标动作

    此处 t 从 0 → 1（0=noise, 1=data）。
    """
    model.eval()
    with torch.no_grad():
        # 取必要字段
        B, W, D = batch["target_motion_local"].shape
        prefix_mask = batch.get("prefix_mask")   # [B, W] bool or None
        gt_motion   = batch["target_motion_local"].to(device)
        seq_mask    = batch.get("seq_mask")

        # 初始噪声
        x = torch.randn(B, W, D, device=device)

        # prefix 写入 GT（推理时 prefix = 上一段末尾观测值）
        if prefix_mask is not None:
            prefix_mask = prefix_mask.to(device)
            x = apply_prefix_mask(x, gt_motion, prefix_mask)

        dt = 1.0 / n_steps
        ts = torch.linspace(0.0, 1.0 - dt, n_steps, device=device)

        for step, t_val in enumerate(ts):
            t_batch = torch.full((B,), t_val, device=device)

            _common = dict(
                self_history_local    = batch.get("self_history_local"),
                partner_history_local = batch.get("partner_history_local"),
                partner_rollout_local = batch.get("partner_rollout_local"),
                partner_time_ids      = batch.get("partner_time_ids"),
                phase_text_encoded    = batch.get("phase_text_encoded"),
                global_text_encoded   = batch.get("global_text_encoded"),
                seq_mask              = seq_mask,
                prefix_mask           = prefix_mask,
                seq_len_ratio         = batch.get("seq_len_ratio"),
            )

            v = model(noisy_motion=x, diffusion_t=t_batch, **_common)   # [B, W, D]

            if method == "euler":
                x = x + dt * v
            elif method == "rk4":
                # RK4：4次 forward（成本高，但精度好）
                k1 = v
                t2 = torch.full((B,), t_val + 0.5 * dt, device=device)
                k2 = model(x + 0.5 * dt * k1, t2, **_common)
                k3 = model(x + 0.5 * dt * k2, t2, **_common)
                t4 = torch.full((B,), t_val + dt, device=device)
                k4 = model(x + dt * k3, t4, **_common)
                x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise ValueError(f"Unknown sampling method: {method}")

            # 每步后强制 prefix 区域不变
            if prefix_mask is not None:
                x = apply_prefix_mask(x, gt_motion, prefix_mask)

    return x
