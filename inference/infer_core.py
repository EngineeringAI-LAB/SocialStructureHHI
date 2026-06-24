"""
infer_core.py
-------------
Executor inference primitives for the online text->motion pipeline (run_inference.py).

Contents:
  - ego<->world coordinate transforms for the 201-dim motion representation
  - sample_ode : flow-matching ODE sampler with a pinned self-history prefix
  - load_executor_model : build HHIPartnerModel and load an executor checkpoint
  - render_html : two-person SMPL viewer (optional, needs HunyuanMotion repo)

Motion layout (201-dim, per frame):
  [0:3]    root translation (x, y, z)
  [3:9]    root orientation, 6D
  [9:135]  21 body-joint rotations, 6D
  [135:201] 22 forward-kinematics joint positions, root-relative
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch

from hhi.hhi_partner_model import HHIPartnerModel
from hhi.hhi_state_machine import _yaw_rotation_matrix

logger = logging.getLogger(__name__)

OVERLAP_N  = 10
FLOOR_SINK = 0.38


# ── HunyuanMotion repo resolution (for backbone + renderer) ───────────────────

def hy_root() -> str:
    """Resolve the HunyuanMotion repo root (HUNYUAN_MOTION_ROOT or ../HY-Motion-1.0)."""
    env = os.environ.get("HUNYUAN_MOTION_ROOT")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "HY-Motion-1.0"))


# ── Coordinate transforms (201-dim) ───────────────────────────────────────────

def ego_to_world_motion(motion_ego: np.ndarray, origin: np.ndarray, R: np.ndarray) -> np.ndarray:
    W   = motion_ego.shape[0]
    out = motion_ego.copy()
    out[:, 0:3] = (R @ motion_ego[:, 0:3].T).T + origin
    fk_ego_abs   = motion_ego[:, 135:201].reshape(W, 22, 3) + motion_ego[:, 0:3][:, None, :]
    fk_world_abs = (R @ fk_ego_abs.reshape(-1, 3).T).T + origin
    out[:, 135:201] = (fk_world_abs.reshape(W, 22, 3) - out[:, 0:3][:, None, :]).reshape(W, 66)
    r6d = motion_ego[:, 3:9]
    c0  = np.stack([r6d[:, 0], r6d[:, 2], r6d[:, 4]], axis=-1)
    c1  = np.stack([r6d[:, 1], r6d[:, 3], r6d[:, 5]], axis=-1)
    rot_world = np.matmul(R[np.newaxis], np.stack([c0, c1, np.cross(c0, c1)], axis=-1))
    out[:, 3] = rot_world[:, 0, 0]; out[:, 4] = rot_world[:, 0, 1]
    out[:, 5] = rot_world[:, 1, 0]; out[:, 6] = rot_world[:, 1, 1]
    out[:, 7] = rot_world[:, 2, 0]; out[:, 8] = rot_world[:, 2, 1]
    return out


def world_to_ego_motion(motion_world: np.ndarray, origin: np.ndarray, R: np.ndarray) -> np.ndarray:
    W    = motion_world.shape[0]
    Rinv = R.T
    out  = motion_world.copy()
    out[:, 0:3] = (Rinv @ (motion_world[:, 0:3] - origin).T).T
    fk_world_abs = motion_world[:, 135:201].reshape(W, 22, 3) + motion_world[:, 0:3][:, None, :]
    fk_ego_abs   = (Rinv @ (fk_world_abs.reshape(-1, 3) - origin).T).T
    out[:, 135:201] = (fk_ego_abs.reshape(W, 22, 3) - out[:, 0:3][:, None, :]).reshape(W, 66)
    r6d = motion_world[:, 3:9]
    c0  = np.stack([r6d[:, 0], r6d[:, 2], r6d[:, 4]], axis=-1)
    c1  = np.stack([r6d[:, 1], r6d[:, 3], r6d[:, 5]], axis=-1)
    rot_ego = np.matmul(Rinv[np.newaxis], np.stack([c0, c1, np.cross(c0, c1)], axis=-1))
    out[:, 3] = rot_ego[:, 0, 0]; out[:, 4] = rot_ego[:, 0, 1]
    out[:, 5] = rot_ego[:, 1, 0]; out[:, 6] = rot_ego[:, 1, 1]
    out[:, 7] = rot_ego[:, 2, 0]; out[:, 8] = rot_ego[:, 2, 1]
    return out


def rot_from_6d(r6d: np.ndarray) -> np.ndarray:
    c0 = r6d[[0, 2, 4]]; c0 = c0 / (np.linalg.norm(c0) + 1e-8)
    c1 = r6d[[1, 3, 5]]; c1 = c1 - np.dot(c1, c0) * c0; c1 /= (np.linalg.norm(c1) + 1e-8)
    return np.stack([c0, c1, np.cross(c0, c1)], axis=1)


def yaw_rotation_matrix(yaw: float) -> np.ndarray:
    return _yaw_rotation_matrix(yaw)


def partner_to_tensor(motion_ego: np.ndarray, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    seq  = torch.from_numpy(motion_ego).unsqueeze(0).to(device)
    mask = torch.ones(1, motion_ego.shape[0], dtype=torch.bool, device=device)
    return seq, mask


# ── ODE sampling (self-history prefix pinned throughout) ─────────────

@torch.no_grad()
def sample_ode(
    model            : HHIPartnerModel,
    ctxt_feat        : Optional[torch.Tensor],
    vtxt_feat        : Optional[torch.Tensor],
    seq_mask_full    : torch.Tensor,          # [1, W+10] bool
    self_history_n   : torch.Tensor,          # [1, 10, 201] normalized
    W                : int,                   # target bucket size (without history)
    partner_local    : Optional[torch.Tensor],
    partner_seq_mask : Optional[torch.Tensor],
    n_steps          : int = 50,
    device           : str = "cuda",
) -> torch.Tensor:                            # [1, W, 201] denormalized
    x = torch.randn(1, W + OVERLAP_N, 201, device=device)
    x[:, :OVERLAP_N, :] = self_history_n
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_vec = torch.full((1,), i * dt, device=device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v = model(
                noisy_motion     = x,
                diffusion_t      = t_vec,
                ctxt_feat        = ctxt_feat,
                vtxt_feat        = vtxt_feat,
                partner_local    = partner_local,
                partner_seq_mask = partner_seq_mask,
                seq_mask         = seq_mask_full,
            )
        x = x + v.float() * dt
        x[:, :OVERLAP_N, :] = self_history_n   # pin history frames throughout
    return model.denormalize(x[:, OVERLAP_N:])   # [1, W, 201]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_executor_model(ckpt_path: str, cfg_json: str, norm_stats_path: str, device: str) -> HHIPartnerModel:
    train_cfg: dict = {}
    if cfg_json and os.path.exists(cfg_json):
        with open(cfg_json) as f:
            train_cfg = json.load(f)

    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt     = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("cfg", {})

    def _cfg(key, default):
        return ckpt_cfg.get(key, train_cfg.get(key, default))

    model = HHIPartnerModel(
        backbone_ckpt_path = None,
        backbone_cfg       = _cfg("backbone_cfg", None),
        lora_rank          = _cfg("lora_rank",  16),
        lora_alpha         = _cfg("lora_alpha", 32.0),
        norm_stats_path    = norm_stats_path,
    )
    missing, unexpected = model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    if missing:    logger.warning("Missing keys: %s", missing)
    if unexpected: logger.warning("Unexpected keys: %s", unexpected)
    model.to(device).eval()
    logger.info("executor loaded (epoch=%s)", ckpt.get("epoch", "?"))
    return model


# ── Rendering (optional; requires HunyuanMotion body model + viewer template) ──

def _fk_min_y_world(motion_world: np.ndarray, seq_mask: np.ndarray) -> float:
    T = int(seq_mask.sum())
    if T == 0: return 0.0
    root_y = motion_world[:T, 1]
    fk_rel = motion_world[:T, 135:201].reshape(T, 22, 3)
    return float((fk_rel[:, :, 1] + root_y[:, None]).min())


def _apply_world_rot_y(motion: np.ndarray, theta: float) -> np.ndarray:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    out  = motion.copy()
    x, z = motion[:, 0].copy(), motion[:, 2].copy()
    out[:, 0] = c*x + s*z; out[:, 2] = -s*x + c*z
    r = motion[:, 3:9]
    c0x, c0z = r[:, 0].copy(), r[:, 4].copy()
    c1x, c1z = r[:, 1].copy(), r[:, 5].copy()
    out[:, 3] = c*c0x + s*c0z; out[:, 4] = c*c1x + s*c1z
    out[:, 5] = r[:, 2];        out[:, 6] = r[:, 3]
    out[:, 7] = -s*c0x + c*c0z; out[:, 8] = -s*c1x + c*c1z
    fk = motion[:, 135:201].reshape(-1, 22, 3).copy()
    fx, fz = fk[:, :, 0].copy(), fk[:, :, 2].copy()
    fk[:, :, 0] = c*fx + s*fz; fk[:, :, 2] = -s*fx + c*fz
    out[:, 135:201] = fk.reshape(-1, 66)
    return out


def _motion201_to_smpl_frames(motion: np.ndarray, person_id: int, y_floor: float) -> list:
    from hymotion.pipeline.body_model import construct_smpl_data_dict
    T      = motion.shape[0]
    transl = motion[:, 0:3].copy(); transl[:, 1] -= (y_floor - FLOOR_SINK)
    rot6d  = np.concatenate([
        motion[:, 3:9].reshape(T, 1, 6),
        motion[:, 9:135].reshape(T, 21, 6),
    ], axis=1)
    smpl = construct_smpl_data_dict(
        torch.from_numpy(rot6d.astype(np.float32)),
        torch.from_numpy(transl.astype(np.float32)),
    )
    Rh, Th, poses, betas = smpl["Rh"], smpl["trans"], smpl["poses"], smpl["betas"]
    return [[{
        "id": person_id, "gender": "neutral",
        "Rh": Rh[f:f+1].tolist(), "Th": Th[f:f+1].tolist(),
        "poses": poses[f:f+1].tolist(), "shapes": betas.tolist(), "opacity": 1.0,
    }] for f in range(T)]


def render_html(p1_world: np.ndarray, p2_world: np.ndarray, seq_mask: np.ndarray,
                caption: str, out_path: str) -> None:
    """Write a standalone two-person SMPL viewer HTML. Needs the HunyuanMotion repo."""
    template_path = os.path.join(hy_root(), "scripts", "gradio", "templates",
                                 "index_wooden_static.html")
    T  = int(seq_mask.sum())
    p1 = p1_world[:T]; p2 = p2_world[:T]

    sep = p2[0, [0, 2]] - p1[0, [0, 2]]
    if np.linalg.norm(sep) > 1e-3:
        theta = -float(np.arctan2(sep[1], sep[0]))
        p1 = _apply_world_rot_y(p1, theta); p2 = _apply_world_rot_y(p2, theta)

    y_floor = min(_fk_min_y_world(m, seq_mask) for m in (p1_world, p2_world))
    p1f = _motion201_to_smpl_frames(p1, 0, y_floor)
    p2f = _motion201_to_smpl_frames(p2, 1, y_floor)
    combined  = [p1f[t] + p2f[t] for t in range(T)]
    smpl_json = json.dumps(combined, ensure_ascii=False)

    caption_html = f'''
    <div class="caption-overlay"><div class="motion-info">
      <div class="captions-section" style="min-width:540px;">
        <div class="caption-item"><b style="color:#FF5700">&#9632; P1</b></div>
        <div class="caption-item"><b style="color:#00C2FF">&#9632; P2</b></div>
        <div class="caption-item" style="font-size:11px;opacity:0.8;white-space:normal;line-height:1.4">{caption}</div>
      </div></div></div>'''

    with open(template_path, encoding="utf-8") as f:
        template = f.read()
    html = template.replace("{{ smpl_data_json }}", smpl_json)
    html = html.replace("{{ caption_html }}", caption_html)
    color_patch = """
                    result.mesh.material = result.mesh.material.clone();
                    var hex = (data.id === 0) ? 0xFF5700 : 0x00C2FF;
                    result.mesh.material.color.setHex(hex);
                    if (result.mesh.material.emissive) {
                        result.mesh.material.emissive.setHex(hex);
                        result.mesh.material.emissiveIntensity = 0.12;
                    }
                    if ('roughness' in result.mesh.material) result.mesh.material.roughness = 1.0;
                    if ('metalness' in result.mesh.material) result.mesh.material.metalness = 0.0;
"""
    html = html.replace("scene.add(result.mesh);", "scene.add(result.mesh);" + color_patch, 1)
    html = html.replace(
        "function computeOffsets(batchSize) {\n        const spacing = 2.0;",
        "function computeOffsets(batchSize) {\n        return Array(batchSize).fill(0);\n        const spacing = 2.0;",
        1,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Saved viewer -> %s", out_path)
