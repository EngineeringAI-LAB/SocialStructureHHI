"""
interhuman_dataset_builder.py
------------------------------
InterHuman (InterGen) dataset adapter.

Converts .pkl files (SMPL params, 59.94 fps) into the same sequence dict
format as hhi_dataset_builder, then delegates all phase segmentation,
annotation, and sample building to the shared pipeline unchanged.

Sequence dict keys (compatible subset of hhi_dataset_builder format):
  seq_id            str              "IH_1042"  (prefix avoids ID collision with Inter-X)
  p1_smplx_params   (T, 22, 3)      SMPL axis-angle joints 0-21
  p2_smplx_params   (T, 22, 3)
  p1_root           (T, 3)          root translation (world)
  p2_root           (T, 3)
  p1_joints         (T, 22, 3)      world-space body joints after FK  (added by compute_world_joints_interhuman)
  p2_joints         (T, 22, 3)
  p1_root_rot       (T, 4)          identity quaternion placeholder
  p2_root_rot       (T, 4)
  global_text       str

Note on joint count: InterHuman uses SMPL (22 body joints only).
_build_hy201_seq reads smplx_params[:, 0:22, :] and joints[:, :22, :],
so (T, 22, 3) arrays work identically to Inter-X's (T, 55, 3) arrays.
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

INTERHUMAN_BASE       = "/scratch3/wan451/3DBody/InterGen/InterHuman Dataset"
INTERHUMAN_MOTIONS    = os.path.join(INTERHUMAN_BASE, "motions")
INTERHUMAN_ANNOTS     = os.path.join(INTERHUMAN_BASE, "annots")
INTERHUMAN_SPLIT_ROOT = os.path.join(INTERHUMAN_BASE, "split")

SRC_FPS = 59.94
TGT_FPS = 30.0

# Prefix to avoid seq_id collision with Inter-X in shared LLM annotation cache
SEQ_ID_PREFIX = "IH_"


# ── Resampling ────────────────────────────────────────────────────────────────

def _resample(arr: np.ndarray, src_fps: float = SRC_FPS, tgt_fps: float = TGT_FPS) -> np.ndarray:
    """Downsample (T, ...) array from src_fps to tgt_fps via nearest-frame selection."""
    T = arr.shape[0]
    if T == 0:
        return arr
    t_src = np.arange(T) / src_fps
    t_tgt = np.arange(0, t_src[-1], 1.0 / tgt_fps)
    idx   = np.round(t_tgt * src_fps).astype(int).clip(0, T - 1)
    return arr[idx]


# ── Load sequence ─────────────────────────────────────────────────────────────

def load_interhuman_sequence(
    seq_id_raw  : str,
    motions_dir : str = INTERHUMAN_MOTIONS,
    annots_dir  : str = INTERHUMAN_ANNOTS,
) -> Dict[str, Any]:
    """
    Load one InterHuman sequence and return a sequence dict compatible with
    hhi_dataset_builder's downstream pipeline.

    seq_id_raw : raw integer ID string, e.g. "1042"
    Returns seq_id = "IH_1042" in the dict.
    """
    pkl_path = os.path.join(motions_dir, f"{seq_id_raw}.pkl")
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    src_fps = float(raw.get("mocap_framerate", SRC_FPS))

    def _extract(person_key: str) -> Dict[str, np.ndarray]:
        p = raw[person_key]
        return {
            "trans"      : _resample(np.array(p["trans"],       dtype=np.float32), src_fps),
            "root_orient": _resample(np.array(p["root_orient"], dtype=np.float32), src_fps),
            "pose_body"  : _resample(np.array(p["pose_body"],   dtype=np.float32), src_fps),
        }

    p1 = _extract("person1")
    p2 = _extract("person2")
    T  = min(len(p1["trans"]), len(p2["trans"]))   # sync to shorter person

    def _make_params(p: Dict[str, np.ndarray]) -> np.ndarray:
        """Build (T, 22, 3) axis-angle array: slot 0 = root_orient, slots 1-21 = pose_body."""
        root = p["root_orient"][:T]               # (T, 3)
        body = p["pose_body"][:T].reshape(T, 21, 3)   # (T, 21, 3)
        return np.concatenate([root[:, None, :], body], axis=1).astype(np.float32)

    # Global text: prefer 2nd description (neutral phrasing), fall back to 1st
    global_text = ""
    annot_path = os.path.join(annots_dir, f"{seq_id_raw}.txt")
    if os.path.exists(annot_path):
        with open(annot_path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        global_text = lines[1] if len(lines) >= 2 else (lines[0] if lines else "")

    return {
        "seq_id"          : SEQ_ID_PREFIX + seq_id_raw,
        "p1_smplx_params" : _make_params(p1),          # (T, 22, 3)
        "p2_smplx_params" : _make_params(p2),          # (T, 22, 3)
        "p1_root"         : p1["trans"][:T].astype(np.float32),
        "p2_root"         : p2["trans"][:T].astype(np.float32),
        "global_text"     : global_text,
    }


# ── FK: SMPL body joints ──────────────────────────────────────────────────────

def compute_world_joints_interhuman(
    sequence   : Dict[str, Any],
    body_model = None,
    device     : str = "cpu",
) -> Dict[str, Any]:
    """
    Run FK on InterHuman SMPL params to get world-space joint positions.
    Adds p1_joints / p2_joints (T, 22, 3) and p1_root_rot / p2_root_rot.

    Reuses the SMPL-X BodyModel from hhi_dataset_builder with zeros for
    hand/face joints (outputs joints 0-21 only).
    """
    try:
        import torch
        from human_body_prior.body_model.body_model import BodyModel
    except ImportError as e:
        raise ImportError(f"FK requires torch + human_body_prior: {e}")

    if body_model is None:
        from .hhi_dataset_builder import _find_smplx_model_path
        bm_path    = os.environ.get("SMPLX_MODEL_PATH", _find_smplx_model_path())
        body_model = BodyModel(bm_fname=bm_path, num_betas=10).to(device)
        body_model.eval()

    def _fk_one(params: np.ndarray, root_trans: np.ndarray) -> np.ndarray:
        """params: (T, 22, 3) axis-angle; returns (T, 22, 3) world joints."""
        T = params.shape[0]
        import torch
        with torch.no_grad():
            out = body_model(
                root_orient = torch.tensor(params[:, 0, :],  dtype=torch.float32, device=device),
                pose_body   = torch.tensor(
                    params[:, 1:22, :].reshape(T, 63),
                    dtype=torch.float32, device=device,
                ),
                pose_hand   = torch.zeros(T, 90, dtype=torch.float32, device=device),
                pose_jaw    = torch.zeros(T,  3, dtype=torch.float32, device=device),
                pose_eye    = torch.zeros(T,  6, dtype=torch.float32, device=device),
                trans       = torch.tensor(root_trans, dtype=torch.float32, device=device),
            )
        joints = out.Jtr.detach().cpu().numpy()[:, :22, :]   # (T, 22, 3)
        # NaN cleanup
        for t in range(joints.shape[0]):
            if np.any(np.isnan(joints[t])):
                joints[t] = joints[t - 1] if t > 0 else 0.0
        return joints.astype(np.float32)

    p1_joints = _fk_one(sequence["p1_smplx_params"], sequence["p1_root"])
    p2_joints = _fk_one(sequence["p2_smplx_params"], sequence["p2_root"])

    T        = p1_joints.shape[0]
    identity = np.tile([0.0, 0.0, 0.0, 1.0], (T, 1)).astype(np.float32)

    sequence["p1_joints"]   = p1_joints
    sequence["p2_joints"]   = p2_joints
    sequence["p1_root_rot"] = identity.copy()
    sequence["p2_root_rot"] = identity.copy()

    # ── Z-up → Y-up coordinate conversion ────────────────────────────────────
    # InterHuman raw data (pkl) stores translations in Z-up space (standing
    # person has trans.z ≈ 0.82m).  The shared ego-centric pipeline and
    # HunyuanMotion both expect Y-up.  Apply the conversion here so that all
    # downstream code (apply_y_grounding_sequence, build_egocentric_transform,
    # _build_hy201_seq) sees a consistent Y-up world.
    #
    # Transform:  x_new = x,   y_new = z,   z_new = -y
    # Matrix (rotation around X by -90°, det = +1):
    #   M = [[1, 0,  0],
    #        [0, 0,  1],
    #        [0,-1,  0]]
    sequence = _apply_zup_to_yup(sequence)

    return sequence


# ── Z-up → Y-up helper (called inside compute_world_joints_interhuman) ────────

_M_ZUP_TO_YUP = np.array([[1.,  0., 0.],
                            [0.,  0., 1.],
                            [0., -1., 0.]], dtype=np.float32)


def _apply_zup_to_yup(sequence: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all world-space vectors in *sequence* from Z-up to Y-up.

    Modifies (in-place and returns):
      p{1,2}_joints      : (T, J, 3) world joint positions
      p{1,2}_root        : (T, 3)    root translations
      p{1,2}_smplx_params[:, 0, :] : root_orient axis-angle (world rotation)

    Body joints 1-21 are parent-relative and are NOT affected.
    """
    from scipy.spatial.transform import Rotation as ScipyR
    M = _M_ZUP_TO_YUP

    for p in ("p1", "p2"):
        # Joint positions: (T, J, 3) → apply M to each 3-vector
        jts = sequence[f"{p}_joints"]
        sequence[f"{p}_joints"] = np.einsum("ij,tnj->tni", M, jts).astype(np.float32)

        # Root translations: (T, 3)
        root = sequence[f"{p}_root"]
        sequence[f"{p}_root"] = (M @ root.T).T.astype(np.float32)

        # Root orientation: convert R_zup → R_yup = M @ R_zup
        params  = sequence[f"{p}_smplx_params"]   # (T, 22, 3)
        root_aa = params[:, 0, :]                  # (T, 3) axis-angle
        R_zup   = ScipyR.from_rotvec(root_aa).as_matrix()          # (T, 3, 3)
        R_yup   = np.einsum("ij,tjk->tik", M, R_zup)                # M @ R
        root_aa_yup = ScipyR.from_matrix(R_yup).as_rotvec().astype(np.float32)
        new_params  = params.copy()
        new_params[:, 0, :] = root_aa_yup
        sequence[f"{p}_smplx_params"] = new_params

    return sequence


# ── Split utilities ───────────────────────────────────────────────────────────

def load_interhuman_split(split: str, split_root: str = INTERHUMAN_SPLIT_ROOT) -> List[str]:
    """Return list of raw seq_id strings (no IH_ prefix) for the given split."""
    path = os.path.join(split_root, f"{split}.txt")
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def action_name_interhuman(_seq_id: str) -> str:
    """InterHuman has no action category; return empty string."""
    return ""
