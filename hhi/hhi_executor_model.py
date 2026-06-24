"""
hhi_executor_model.py (v2 - LoRA + backbone.forward)
------------------------------------------------------
HunyuanMotion backbone 全参数冻结 + LoRA 微调。
条件输入：仅 phase action text（Qwen3-8B ctxt + CLIP vtxt）。
移除所有 HHI 特有条件（partner rollout、history、schema、role/mode embed 等）。

LoRA 注入位置（double_blocks 和 single_blocks 的注意力投影层）：
  double_blocks[*].motion_qkv       [feat_dim → feat_dim*3]
  double_blocks[*].motion_out_proj   [feat_dim → feat_dim]
  double_blocks[*].text_qkv         [feat_dim → feat_dim*3]
  double_blocks[*].text_out_proj     [feat_dim → feat_dim]
  single_blocks[*].linear1           [feat_dim → feat_dim*3 + mlp_hidden]
  single_blocks[*].linear2           [feat_dim + mlp_hidden → feat_dim]
"""
from __future__ import annotations

import logging
import math
import os
import sys
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_HY_ROOT = os.environ.get(
    "HUNYUAN_MOTION_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "..", "HY-Motion-1.0"),
)

_HY_DEFAULT_CFG = dict(
    input_dim      = 201,
    feat_dim       = 1024,
    output_dim     = 201,
    ctxt_input_dim = 4096,
    vtxt_input_dim = 768,
    num_layers     = 18,
    num_heads      = 16,
    mlp_ratio      = 4.0,
    mlp_act_type   = "gelu_tanh",
    qk_norm_type   = "rms",
    qkv_bias       = True,
    dropout        = 0.0,
    mask_mode      = "narrowband",
    apply_rope_to_single_branch = False,
    time_factor    = 1000.0,
    narrowband_length = 2.0,
)


def _import_hunyuan():
    hy_root = os.path.abspath(_HY_ROOT)
    if hy_root not in sys.path:
        sys.path.insert(0, hy_root)
    from hymotion.network.hymotion_mmdit import HunyuanMotionMMDiT
    return HunyuanMotionMMDiT


# ── LoRA ──────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """将冻结的 Linear 层替换为 frozen_original + trainable low-rank delta。

    output = W(x) + scale * B(A(x))
      A: [in_features → rank]  (kaiming init)
      B: [rank → out_features] (zero init)
    """

    def __init__(self, original: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.original = original          # frozen
        in_f  = original.in_features
        out_f = original.out_features
        self.lora_A = nn.Linear(in_f,  rank,  bias=False)
        self.lora_B = nn.Linear(rank,  out_f, bias=False)
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + self.scale * self.lora_B(self.lora_A(x))


# ── HHIExecutorModel ──────────────────────────────────────────────────────────

class HHIExecutorModel(nn.Module):
    """
    HunyuanMotion Lite backbone (全冻结) + LoRA 微调。

    可训练参数仅为 LoRA 的 A/B 矩阵（~8M），约为 backbone 总参数的 1.7%。

    forward() 直接调用 backbone.forward()，输入：
      noisy_motion : [B, W, 201]  归一化噪声动作
      diffusion_t  : [B]          flow matching 时间步
      ctxt_feat    : [B, T, 4096] Qwen3 phase text 特征（预编码）
      vtxt_feat    : [B, 1, 768]  CLIP phase text 特征（预编码）
      seq_mask     : [B, W] bool  True=有效帧
    返回：
      pred_velocity: [B, W, 201]  预测速度场
    """

    def __init__(
        self,
        backbone_ckpt_path : Optional[str] = None,
        backbone_cfg       : Optional[dict] = None,
        lora_rank          : int   = 16,
        lora_alpha         : float = 32.0,
        norm_stats_path    : Optional[str] = None,
    ) -> None:
        super().__init__()

        HunyuanMotionMMDiT = _import_hunyuan()
        cfg = dict(_HY_DEFAULT_CFG)
        if backbone_cfg:
            cfg.update(backbone_cfg)
        self.backbone = HunyuanMotionMMDiT(**cfg)
        self._feat_dim = cfg["feat_dim"]

        # 归一化统计量（我们数据集的 mean/std，覆盖 backbone 自带的）
        dim = cfg["input_dim"]
        self.register_buffer("motion_mean",   torch.zeros(1, 1, dim))
        self.register_buffer("motion_std",    torch.ones( 1, 1, dim))
        self.register_buffer("rel_root_std",  torch.ones(3))   # GT 相对 root 位移 std [3]
        self.register_buffer("rel_joint_std", torch.ones(3))   # GT 相对 joint 1-21 位置 std [3]

        # 加载预训练 backbone 权重
        if backbone_ckpt_path and os.path.exists(backbone_ckpt_path):
            self._load_backbone_ckpt(backbone_ckpt_path)
        else:
            logger.warning("backbone_ckpt_path 未提供或不存在: %s", backbone_ckpt_path)

        # 加载我们数据集的归一化统计量（覆盖 backbone 自带的）
        if norm_stats_path and os.path.exists(norm_stats_path):
            import numpy as np
            ns = np.load(norm_stats_path)
            mean = torch.tensor(ns["mean"], dtype=torch.float32).reshape(1, 1, -1)
            std  = torch.tensor(ns["std"],  dtype=torch.float32).reshape(1, 1, -1).clamp(min=1e-3)
            self.motion_mean.copy_(mean)
            self.motion_std.copy_(std)
            if "rel_root_std" in ns:
                self.rel_root_std.copy_(
                    torch.tensor(ns["rel_root_std"], dtype=torch.float32).clamp(min=1e-6)
                )
            if "rel_joint_std" in ns:
                self.rel_joint_std.copy_(
                    torch.tensor(ns["rel_joint_std"], dtype=torch.float32).clamp(min=1e-6)
                )
            logger.info("norm stats loaded from %s (dim=%d)", norm_stats_path, mean.shape[-1])

        # 冻结所有 backbone 参数
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # 注入 LoRA（可训练）
        self._apply_lora(lora_rank, lora_alpha)
        logger.info(
            "LoRA applied (rank=%d alpha=%.1f) | trainable: %dM / backbone: %dM",
            lora_rank, lora_alpha,
            self.count_trainable_params() // 1_000_000,
            self.count_backbone_params()  // 1_000_000,
        )

    # ── LoRA injection ────────────────────────────────────────────────────────

    def _apply_lora(self, rank: int, alpha: float) -> None:
        """替换 attention 投影层为 LoRALinear。"""
        for blk in self.backbone.double_blocks:
            blk.motion_qkv      = LoRALinear(blk.motion_qkv,      rank, alpha)
            blk.motion_out_proj = LoRALinear(blk.motion_out_proj,  rank, alpha)
            blk.text_qkv        = LoRALinear(blk.text_qkv,        rank, alpha)
            blk.text_out_proj   = LoRALinear(blk.text_out_proj,   rank, alpha)
        for blk in self.backbone.single_blocks:
            blk.linear1 = LoRALinear(blk.linear1, rank, alpha)
            blk.linear2 = LoRALinear(blk.linear2, rank, alpha)

    # ── Checkpoint loading ────────────────────────────────────────────────────

    def _load_backbone_ckpt(self, ckpt_path: str) -> None:
        logger.info("Loading backbone from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = (
            ckpt.get("state_dict") or ckpt.get("model") or
            ckpt.get("model_state_dict") or ckpt
        ) if isinstance(ckpt, dict) else ckpt

        cleaned = {}
        for k, v in state.items():
            k = k.replace("module.", "")
            k = k.removeprefix("backbone.")
            k = k.removeprefix("motion_transformer.")
            cleaned[k] = v

        _KNOWN_UNUSED = {"null_vtxt_feat", "null_ctxt_input",
                         "special_game_vtxt_feat", "special_game_ctxt_feat", "mean", "std"}
        if "mean" in cleaned and "std" in cleaned:
            mean = cleaned["mean"].float().reshape(1, 1, -1)
            std  = cleaned["std"].float().reshape(1, 1, -1).clamp(min=1e-3)
            self.motion_mean.copy_(mean)
            self.motion_std.copy_(std)

        missing, unexpected = self.backbone.load_state_dict(cleaned, strict=False)
        if missing:
            logger.warning("backbone missing keys (%d): %s...", len(missing), missing[:3])
        unexpected_unknown = [k for k in unexpected if k not in _KNOWN_UNUSED]
        if unexpected_unknown:
            logger.warning("backbone unexpected keys (%d): %s...",
                           len(unexpected_unknown), unexpected_unknown[:3])
        logger.info("backbone loaded successfully")

    # ── Normalization ─────────────────────────────────────────────────────────

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.motion_mean) / self.motion_std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.motion_std + self.motion_mean

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        noisy_motion : torch.Tensor,            # [B, W, 201] 归一化
        diffusion_t  : torch.Tensor,            # [B]
        ctxt_feat    : Optional[torch.Tensor],  # [B, T, 4096] Qwen 特征
        vtxt_feat    : Optional[torch.Tensor],  # [B, 1, 768]  CLIP 特征
        seq_mask     : Optional[torch.BoolTensor] = None,  # [B, W]
    ) -> torch.Tensor:
        B, W, _ = noisy_motion.shape
        device   = noisy_motion.device

        # ctxt_mask: 全 True（预编码特征已 padding 到固定长度）
        if ctxt_feat is not None:
            ctxt_mask = torch.ones(B, ctxt_feat.shape[1], dtype=torch.bool, device=device)
        else:
            # fallback：空文本（1 token）
            ctxt_feat = torch.zeros(B, 1, 4096, device=device, dtype=noisy_motion.dtype)
            ctxt_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

        if vtxt_feat is None:
            vtxt_feat = torch.zeros(B, 1, 768, device=device, dtype=noisy_motion.dtype)

        if seq_mask is None:
            seq_mask = torch.ones(B, W, dtype=torch.bool, device=device)

        pred = self.backbone(
            x                = noisy_motion,
            ctxt_input       = ctxt_feat.float(),
            vtxt_input       = vtxt_feat.float(),
            timesteps        = diffusion_t,
            x_mask_temporal  = seq_mask,
            ctxt_mask_temporal = ctxt_mask,
        )
        return pred   # [B, W, 201]

    # ── train() override ──────────────────────────────────────────────────────

    def train(self, mode: bool = True):
        """保持 backbone 大部分处于 eval 模式（Dropout=0.0，影响不大，但保持一致）。"""
        super().train(mode)
        # LoRALinear 内的 original 是 frozen，让它 eval
        if mode:
            for p in self.backbone.parameters():
                if not p.requires_grad:
                    pass  # frozen params 不受 train/eval 影响（dropout=0）
        return self

    # ── Parameter helpers ─────────────────────────────────────────────────────

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_backbone_params(self) -> int:
        return sum(p.numel() for p in self.backbone.parameters())

    # ── Post-processing smoothing ─────────────────────────────────────────────

    @staticmethod
    def smooth_motion(motion: torch.Tensor) -> torch.Tensor:
        """Apply HunyuanMotion-style post-processing smoothing to denormalized motion.

        Args:
            motion: [T, 201] or [B, T, 201] denormalized motion tensor.

        Returns:
            Smoothed tensor of same shape.

        Layout of 201-dim:
            [0:3]    root translation (savgol)
            [3:135]  rot6d for 22 joints (slerp), shape [*, 22, 6]
            [135:201] FK joint positions (savgol)
        """
        hy_root = os.path.abspath(_HY_ROOT)
        if hy_root not in sys.path:
            sys.path.insert(0, hy_root)
        from hymotion.pipeline.motion_diffusion import MotionGeneration

        squeeze = motion.ndim == 2
        if squeeze:
            motion = motion.unsqueeze(0)   # [1, T, 201]

        B, T, D = motion.shape
        out = motion.clone()

        # ── Translation: dims [0:3] ──
        transl = motion[:, :, 0:3]                        # [B, T, 3]
        out[:, :, 0:3] = MotionGeneration.smooth_with_savgol(
            transl, window_length=11, polyorder=5
        )

        # ── Rotations: dims [3:135] → [B, T, 22, 6] ──
        rot6d = motion[:, :, 3:135].reshape(B, T, 22, 6)
        out[:, :, 3:135] = MotionGeneration.smooth_with_slerp(
            rot6d, sigma=1.0
        ).reshape(B, T, 132)

        # ── FK positions: dims [135:201] → [B, T, 22, 3] ──
        fk = motion[:, :, 135:201].reshape(B, T, 66)      # treat as [B, T, 66]
        # savgol expects [B, T, C]; reshape to [B, T, 66] already works
        out[:, :, 135:201] = MotionGeneration.smooth_with_savgol(
            fk, window_length=11, polyorder=5
        )

        if squeeze:
            out = out.squeeze(0)
        return out
