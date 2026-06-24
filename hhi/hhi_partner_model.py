"""
hhi_partner_model.py
--------------------
HHIExecutorModel 的子类，增加变长 partner 序列条件。

新增组件（可训练）：
  partner_proj  : Linear(201 → 4096, no bias)  将 partner 运动投影到 ctxt 空间
  null_partner  : Parameter [1, 1, 4096]        推理 CFG unconditional 时使用

forward() 新增参数：
  partner_local    : [B, W_p, 201]  partner 运动（原始未归一化，W_p 可 ≠ W）
  partner_seq_mask : [B, W_p] bool  True=有效帧（全 False 等价于无 partner）

训练时 CFG dropout：
  将对应样本的 partner_seq_mask 置全 False（backbone cross-attn 不再使用 partner tokens）。

实现：
  partner tokens 归一化→投影后拼接到 ctxt_feat 末尾，送入 backbone cross-attention。
"""
from __future__ import annotations

import math
import logging
from typing import Optional

import torch
import torch.nn as nn

from .hhi_executor_model import HHIExecutorModel, _HY_DEFAULT_CFG

logger = logging.getLogger(__name__)


class HHIPartnerModel(HHIExecutorModel):
    """HHIExecutorModel + 变长 partner 运动条件。"""

    def __init__(
        self,
        backbone_ckpt_path : Optional[str] = None,
        backbone_cfg       : Optional[dict] = None,
        lora_rank          : int   = 16,
        lora_alpha         : float = 32.0,
        norm_stats_path    : Optional[str] = None,
    ) -> None:
        super().__init__(
            backbone_ckpt_path = backbone_ckpt_path,
            backbone_cfg       = backbone_cfg,
            lora_rank          = lora_rank,
            lora_alpha         = lora_alpha,
            norm_stats_path    = norm_stats_path,
        )

        cfg      = dict(_HY_DEFAULT_CFG)
        if backbone_cfg:
            cfg.update(backbone_cfg)
        ctxt_dim = cfg["ctxt_input_dim"]   # 4096
        in_dim   = cfg["input_dim"]        # 201

        # partner 投影层（可训练）
        self.partner_proj = nn.Linear(in_dim, ctxt_dim, bias=False)
        nn.init.normal_(self.partner_proj.weight, std=0.02)

        # null token：推理时 unconditional pass 用（全 False mask 时 backbone 不看，但占位）
        self.null_partner = nn.Parameter(torch.zeros(1, 1, ctxt_dim))

        logger.info(
            "HHIPartnerModel: partner_proj(%d→%d) | trainable: %dM / backbone: %dM",
            in_dim, ctxt_dim,
            self.count_trainable_params() // 1_000_000,
            self.count_backbone_params()  // 1_000_000,
        )

    def forward(
        self,
        noisy_motion     : torch.Tensor,                     # [B, W, 201]
        diffusion_t      : torch.Tensor,                     # [B]
        ctxt_feat        : Optional[torch.Tensor],           # [B, T, 4096]
        vtxt_feat        : Optional[torch.Tensor],           # [B, 1, 768]
        partner_local    : Optional[torch.Tensor] = None,    # [B, W_p, 201] 原始空间
        partner_seq_mask : Optional[torch.BoolTensor] = None,# [B, W_p] True=有效帧
        seq_mask         : Optional[torch.BoolTensor] = None,# [B, W]
    ) -> torch.Tensor:
        B, W, _ = noisy_motion.shape
        device  = noisy_motion.device
        dtype   = noisy_motion.dtype

        # ── Text context ─────────────────────────────────────────────────────
        if ctxt_feat is not None:
            ctxt_mask = torch.ones(B, ctxt_feat.shape[1], dtype=torch.bool, device=device)
        else:
            ctxt_feat = torch.zeros(B, 1, 4096, device=device, dtype=dtype)
            ctxt_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

        if vtxt_feat is None:
            vtxt_feat = torch.zeros(B, 1, 768, device=device, dtype=dtype)

        if seq_mask is None:
            seq_mask = torch.ones(B, W, dtype=torch.bool, device=device)

        # ── Partner tokens ────────────────────────────────────────────────────
        if partner_local is not None:
            W_p = partner_local.shape[1]
            partner_n   = self.normalize(partner_local.to(device))
            partner_tok = self.partner_proj(partner_n.to(dtype))     # [B, W_p, 4096]

            if partner_seq_mask is not None:
                p_mask = partner_seq_mask.to(device=device)           # [B, W_p]
            else:
                p_mask = torch.ones(B, W_p, dtype=torch.bool, device=device)
        else:
            # 无 partner：用 null token（1 帧，mask=False → backbone 不使用）
            W_p         = 1
            partner_tok = self.null_partner.to(dtype=dtype, device=device).expand(B, -1, -1)
            p_mask      = torch.zeros(B, 1, dtype=torch.bool, device=device)

        ctxt_feat = torch.cat([ctxt_feat.to(dtype), partner_tok], dim=1)  # [B, T+W_p, 4096]
        ctxt_mask = torch.cat([ctxt_mask, p_mask], dim=1)                 # [B, T+W_p]

        pred = self.backbone(
            x                  = noisy_motion,
            ctxt_input         = ctxt_feat.float(),
            vtxt_input         = vtxt_feat.float(),
            timesteps          = diffusion_t,
            x_mask_temporal    = seq_mask,
            ctxt_mask_temporal = ctxt_mask,
        )
        return pred   # [B, W, 201]
