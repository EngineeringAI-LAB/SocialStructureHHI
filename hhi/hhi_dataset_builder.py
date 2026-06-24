"""
hhi_dataset_builder.py
-----------------------
GT 数据回放、状态机打标、phase 切分、Ego-centric 变换、训练样本导出。

核心函数：
  replay_and_label_modes()
  segment_interaction_phases()
  build_phase_sample()
  export_dataset()

依赖：hhi_constants, hhi_types, hhi_schema, hhi_duration_buckets, hhi_state_machine
"""
from __future__ import annotations

import json
import logging
import os
import re as _re
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from .hhi_constants import (
    CONTACT_DIST_THRESH,
    ENTRY_DIST,
    EXIT_DIST,
    INTERX_ACTION_NAMES,
    INTERX_H5_PATH,
    INTERX_SPLIT_ROOT,
    INTERX_TEXT_DIR,
    MIN_CONTACT_FRAMES,
    MIN_PHASE_FRAMES,
    NON_CONTACT_WRIST_SPEED,
    OVERLAP_N,
    REACH_DIST_THRESH,
    REACH_WINDUP_FRAMES,
    RELEASE_DIST_THRESH,
    W_PREV,
)
from .hhi_duration_buckets import _nearest_bucket, bucketize_phase_sequence, split_phase_if_needed

from .hhi_state_machine import (
    HysteresisStateMachine,
    apply_egocentric_transform,
    apply_y_grounding_sequence,
    build_egocentric_transform,
    _extract_root_yaw,
    _yaw_rotation_matrix,
)
from .hhi_types import (
    InteractionMode,
    NA,
    PhaseSample,
    PartnerSource,
)

logger = logging.getLogger(__name__)


# ── Sequence ID helpers ───────────────────────────────────────────────────────

def _action_name_from_seq_id(seq_id: str) -> str:
    """Parse action category name from sequence ID (e.g. 'G001T000A005R000' → 'Kick')."""
    m = _re.search(r"A(\d+)", seq_id)
    if not m:
        return ""
    idx = int(m.group(1))
    return INTERX_ACTION_NAMES[idx] if idx < len(INTERX_ACTION_NAMES) else ""


# 4-type canonical phase schema
_CONTACT_PHASE_TYPES    = {"contact"}
_PRECONTACT_PHASE_TYPES = {"approach"}
_POSTCONTACT_PHASE_TYPES = {"release"}

# 内部状态机（6类）→ 对外 canonical（4类）
_INTERNAL_TO_CANONICAL: dict = {
    "approach":             "approach",
    "reach":                "approach",
    "contact_hold":         "contact",
    "release":              "release",
    "step_back":            "release",
    "non_contact_interact": "non_contact_interact",
}


def _validate_phase_type(
    phase_type       : str,
    has_contact      : bool,
    is_first_contact : bool = False,
    dist_delta       : Optional[float] = None,   # root 间距变化量（正=分离，负=接近）
    seen_contact     : bool = False,              # 序列中是否已发生过接触
) -> str:
    """
    用物理检测结果校正 LLM 标注的 phase_type（4-type schema）。

    规则（优先级从高到低）：
      1. has_contact=True  → 必须是 contact
      2. has_contact=False，但标了 contact → 修正为 approach
      3. has_contact=False，已发生过接触，dist_delta>0 → 应是 release
      4. 其余保持原值
    """
    # 先规范化旧标签（兼容已有缓存中的旧类型）
    phase_type = _INTERNAL_TO_CANONICAL.get(phase_type, phase_type)

    if has_contact:
        if phase_type not in _CONTACT_PHASE_TYPES:
            logger.debug("phase_type=%r 与 has_contact=True 矛盾，修正为 contact", phase_type)
            return "contact"
        return phase_type

    # 无接触段
    if phase_type in _CONTACT_PHASE_TYPES:
        logger.debug("phase_type=%r 与 has_contact=False 矛盾，修正为 approach", phase_type)
        return "approach"

    # 接触结束后若 root 间距在增大，应是 release
    if seen_contact and dist_delta is not None and dist_delta > 0.05:
        if phase_type in _PRECONTACT_PHASE_TYPES:
            logger.debug(
                "phase_type=%r 但接触后 dist_delta=%.2f>0，修正为 release",
                phase_type, dist_delta,
            )
            return "release"

    return phase_type


# ── Split file loading ────────────────────────────────────────────────────────

def load_split_ids(split_name: str, split_root: str = INTERX_SPLIT_ROOT) -> List[str]:
    """读取 train.txt / val.txt / test.txt；返回 sequence id 列表。"""
    path = os.path.join(split_root, f"{split_name}.txt")
    with open(path, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


# ── H5 data loading ───────────────────────────────────────────────────────────

def load_sequence_from_h5(
    seq_id   : str,
    h5_path  : str = INTERX_H5_PATH,
    text_dir : str = INTERX_TEXT_DIR,
) -> Dict[str, Any]:
    """
    从 inter-x_regen.h5 读取单条双人序列。

    实际 H5 布局：f[seq_id] 是 Dataset，shape=(T, 56, 6)
      dim1: 56 SMPL-X joints；joint 55 (TRANSL_IDX) = root 平移
      dim2: 6 channels，前3 = P1 axis-angle，后3 = P2 axis-angle

    返回 dict：
      p1_smplx_params : (T, 56, 3) float32  P1 SMPL-X axis-angle rotations
      p2_smplx_params : (T, 56, 3) float32  P2 SMPL-X axis-angle rotations
      p1_root         : (T, 3)     float32  P1 root translation (world)
      p2_root         : (T, 3)     float32  P2 root translation (world)
      global_text     : str
      seq_id          : str

    注：world-space joints 需要 FK（SMPL-X body model），由
    compute_world_joints() 单独计算。
    """
    TRANSL_IDX = 55   # joint index for root translation in Inter-X H5

    with h5py.File(h5_path, "r") as f:
        motion = f[seq_id][:]   # (T, 56, 6)

    p1_params = motion[:, :, 0:3]        # (T, 56, 3) P1 axis-angle
    p2_params = motion[:, :, 3:6]        # (T, 56, 3) P2 axis-angle
    p1_root   = motion[:, TRANSL_IDX, 0:3]  # (T, 3) P1 translation
    p2_root   = motion[:, TRANSL_IDX, 3:6]  # (T, 3) P2 translation

    # Text from separate .txt file (3 paraphrase descriptions).
    # Use the 2nd non-empty line: neutral "a person / the other person" phrasing,
    # avoids first/second person ambiguity. Fall back to 1st line if needed.
    global_text = ""
    text_path = os.path.join(text_dir, f"{seq_id}.txt")
    if os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8") as tf:
            non_empty_lines = [l.strip() for l in tf if l.strip()]
        if len(non_empty_lines) >= 2:
            global_text = non_empty_lines[1]
        elif non_empty_lines:
            global_text = non_empty_lines[0]

    return {
        "seq_id"          : seq_id,
        "p1_smplx_params" : p1_params.astype(np.float32),
        "p2_smplx_params" : p2_params.astype(np.float32),
        "p1_root"         : p1_root.astype(np.float32),
        "p2_root"         : p2_root.astype(np.float32),
        "global_text"     : global_text,
    }


# ── FK: SMPL-X → world joints ────────────────────────────────────────────────

_SMPLX_NEUTRAL_CANDIDATES = [
    "/scratch3/wan451/3DBody/HHILLM/body_models/smplx/SMPLX_NEUTRAL.npz",
    "/scratch3/wan451/3DBody/HHILLM/body_models/smplx/SMPLX_NEUTRAL_2020.npz",
    "/scratch3/wan451/3DBody/Duolando/smplx/SMPLX_NEUTRAL.npz",
]

_SMPLX_JOINT_INDICES = {
    "ROOT_IDX": 0, "BODY_START": 1, "BODY_END": 22,
    "JAW_IDX": 22, "LEYE_IDX": 23, "REYE_IDX": 24,
    "LHAND_START": 25, "LHAND_END": 40,
    "RHAND_START": 40, "RHAND_END": 55,
    "TRANSL_IDX": 55,
}


def _find_smplx_model_path() -> str:
    for path in _SMPLX_NEUTRAL_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Cannot find SMPL-X neutral model. Set SMPLX_MODEL_PATH env var or ensure one of "
        f"{_SMPLX_NEUTRAL_CANDIDATES} exists."
    )


def _run_fk_for_person(body_model, motion: np.ndarray, person_offset: int, device) -> np.ndarray:
    """
    Run FK for one person in a (T, 56, 6) SMPL-X motion array.
    person_offset=0 → P1 (channels 0:3), person_offset=3 → P2 (channels 3:6).
    Returns (T, 55, 3) world-space joints.
    """
    import torch
    i = _SMPLX_JOINT_INDICES
    pose_body = torch.tensor(
        motion[:, i["BODY_START"]:i["BODY_END"], person_offset:person_offset + 3].reshape(len(motion), -1),
        dtype=torch.float32, device=device,
    )
    pose_hand = torch.tensor(
        np.concatenate([
            motion[:, i["LHAND_START"]:i["LHAND_END"], person_offset:person_offset + 3],
            motion[:, i["RHAND_START"]:i["RHAND_END"], person_offset:person_offset + 3],
        ], axis=1).reshape(len(motion), -1),
        dtype=torch.float32, device=device,
    )
    root_orient = torch.tensor(
        motion[:, i["ROOT_IDX"], person_offset:person_offset + 3],
        dtype=torch.float32, device=device,
    )
    pose_jaw = torch.tensor(
        motion[:, i["JAW_IDX"], person_offset:person_offset + 3],
        dtype=torch.float32, device=device,
    )
    pose_eye = torch.tensor(
        np.concatenate([
            motion[:, i["LEYE_IDX"], person_offset:person_offset + 3],
            motion[:, i["REYE_IDX"], person_offset:person_offset + 3],
        ], axis=1),
        dtype=torch.float32, device=device,
    )
    trans = torch.tensor(
        motion[:, i["TRANSL_IDX"], person_offset:person_offset + 3],
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        body = body_model(
            root_orient=root_orient, pose_body=pose_body,
            pose_hand=pose_hand, pose_jaw=pose_jaw,
            pose_eye=pose_eye, trans=trans,
        )
    return body.Jtr.detach().cpu().numpy()   # (T, 55, 3)


def compute_world_joints(
    sequence   : Dict[str, Any],
    body_model = None,
    device     : str = "cpu",
) -> Dict[str, Any]:
    """
    FK: sequence["p1_smplx_params"] / sequence["p2_smplx_params"] (T, 56, 3)
    → 添加 world-space joints 并回写 sequence：
        p1_joints    : (T, 55, 3)
        p2_joints    : (T, 55, 3)
        p1_root_rot  : (T, 4) quaternion xyzw  (identity placeholder)
        p2_root_rot  : (T, 4)

    若 body_model 为 None，尝试自动加载。
    body_model 可为 human_body_prior.BodyModel 实例或 None。
    """
    try:
        import torch
        from human_body_prior.body_model.body_model import BodyModel
    except ImportError as e:
        raise ImportError(f"FK requires torch + human_body_prior: {e}")

    if body_model is None:
        bm_path = os.environ.get("SMPLX_MODEL_PATH", _find_smplx_model_path())
        body_model = BodyModel(bm_fname=bm_path, num_betas=10).to(device)
        body_model.eval()

    # Reconstruct (T, 56, 6) from stored (T, 56, 3) per-person params
    p1 = sequence["p1_smplx_params"]   # (T, 56, 3)
    p2 = sequence["p2_smplx_params"]   # (T, 56, 3)
    motion = np.concatenate([p1, p2], axis=2)  # (T, 56, 6)

    p1_joints = _run_fk_for_person(body_model, motion, 0, device)  # (T, 55, 3)
    p2_joints = _run_fk_for_person(body_model, motion, 3, device)  # (T, 55, 3)

    # 清理 NaN: 若某帧有 NaN，用前一帧替代（或用平均值）
    for joints in [p1_joints, p2_joints]:
        for t in range(joints.shape[0]):
            if np.any(np.isnan(joints[t])):
                if t > 0:
                    joints[t] = joints[t - 1]  # Use previous frame
                else:
                    joints[t] = 0.0  # Use zero as fallback

    T = p1_joints.shape[0]
    identity = np.tile([0.0, 0.0, 0.0, 1.0], (T, 1)).astype(np.float32)

    sequence["p1_joints"]   = p1_joints.astype(np.float32)
    sequence["p2_joints"]   = p2_joints.astype(np.float32)
    sequence["p1_root_rot"] = identity.copy()
    sequence["p2_root_rot"] = identity.copy()
    return sequence


# ── Step 2: replay & label modes ─────────────────────────────────────────────

def replay_and_label_modes(
    sequence        : Dict[str, Any],
    foot_joint_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, HysteresisStateMachine]:
    """
    对整条序列做前向回放，用迟滞状态机标记每帧 mode。

    返回：
      mode_labels : np.ndarray [T]   dtype=object  (InteractionMode)
      sm          : HysteresisStateMachine  (含 entry/exit 事件日志)
    """
    sm = HysteresisStateMachine(foot_joint_indices=foot_joint_indices or [])
    p1_roots = sequence["p1_root"]
    p2_roots = sequence["p2_root"]
    labels   = sm.replay_sequence(p1_roots, p2_roots)
    return labels, sm


# ── Step 3: segment interaction phases ───────────────────────────────────────

def segment_interaction_phases(
    sequence         : Dict[str, Any],
    mode_labels      : np.ndarray,
    annotation_path  : Optional[str] = None,
    cache_path       : Optional[str] = None,
    keyword_map_path : Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    从已打标的序列中切分交互 phase，返回 phase_meta 列表。

    优先策略：
      1. 用 Inter-X 官方标注段边界（annotation_path）
      2. fallback 到距离驱动切分
    然后统一经过 bucketize_phase_sequence() 合法化。

    每个 phase_meta 包含：
      phase_id, t_start, t_end, raw_duration, duration_bucket,
      global_text, has_contact, is_splittable,
      fallback_reason (若使用 fallback)
    """
    T      = len(mode_labels)
    seq_id = sequence["seq_id"]
    candidates: List[Dict[str, Any]] = []

    # ── 1. 尝试读取官方标注 ──────────────────────────────────────────────────
    annotations = _load_interx_annotations(annotation_path, seq_id)

    if annotations:
        raw_candidates = _build_candidates_from_annotations(
            annotations, mode_labels, sequence,
        )
    else:
        logger.debug("seq %s: 无标注文件，使用距离驱动 fallback", seq_id)
        raw_candidates = _build_candidates_from_distance(mode_labels, sequence)
        for c in raw_candidates:
            c["fallback_reason"] = "no_annotation"

    # ── 2. 物理验证与合并 ────────────────────────────────────────────────────
    merged = _merge_adjacent_phases(raw_candidates, sequence)

    # ── 3. 状态机二次过滤（确保整段处于 INTERACTION_MODE）────────────────────
    # 五状态机产生的候选（hint_phase_type 非空）已经做了正确的语义切分；
    # 其相位边界与 INTERACTION_MODE 门控边界可能因阈值不同而有偏差
    # （接触检测用手腕距离，门控用根节点距离），因此只做宽松过滤：
    #   - approach/reach：无需 INTERACTION_MODE 帧（接近段本就在进入前）
    #   - 其他 hint：只需任意一帧在 INTERACTION_MODE
    # 无 hint 的候选（标注驱动）保持严格过滤（全帧在 INTERACTION_MODE）。
    has_any_interaction = any(
        mode_labels[t] == InteractionMode.INTERACTION for t in range(len(mode_labels))
    )
    _PRE_CONTACT_HINTS = {"approach", "reach"}
    filtered = []
    for p in merged:
        hint = p.get("hint_phase_type", "")
        if not has_any_interaction:
            # 整段序列都是 NAVIGATION_MODE（Wave/Chat 等远距离动作）：
            # 跳过 INTERACTION_MODE 检查，保留所有 phases
            filtered.append(p)
        elif hint:
            # 五状态机产生的所有候选（approach/reach/contact_hold/release/step_back）：
            # 完全信任五状态机的分段，跳过 INTERACTION_MODE 检查。
            # 五状态机使用手腕距离+接触检测，精度高于根节点门控，
            # 接触相位可能发生在 INTERACTION_MODE 入口之前。
            filtered.append(p)
        else:
            # 无 hint（标注驱动 fallback）：严格要求全帧在 INTERACTION_MODE
            if _all_in_interaction(p, mode_labels):
                filtered.append(p)
    if len(filtered) < len(merged):
        logger.info(
            "seq %s: 状态机二次过滤后 %d/%d phases 保留",
            seq_id, len(filtered), len(merged),
        )

    # ── 3.5 首帧扩展 / 序列丢弃规则 ─────────────────────────────────────────
    # 第一个 phase 的 t_start 距序列首帧 (frame 0)：
    #   < 30 帧 → 扩展到 frame 0（保留接近段开头的接近过程）
    #   > 90 帧 → 丢弃整个序列（有效交互出现太晚）
    if filtered:
        first_t = filtered[0]["t_start"]
        if first_t > 90:
            logger.info(
                "seq %s: first phase starts at frame %d (>90), discarding sequence",
                seq_id, first_t,
            )
            return []
        if 0 < first_t < 30:
            logger.debug(
                "seq %s: extending first phase t_start %d → 0",
                seq_id, first_t,
            )
            filtered[0]["t_start"] = 0

    # ── 4. 分配 phase_id ─────────────────────────────────────────────────────
    for i, p in enumerate(filtered):
        p.setdefault("phase_id", f"{seq_id}_phase{i:03d}")
        p["raw_duration"] = p["t_end"] - p["t_start"]

    # ── 5. 时长合法化 ────────────────────────────────────────────────────────
    bucketized, records = bucketize_phase_sequence(filtered)

    # ── 6. Ping-Pong executor 分配 ────────────────────────────────────────────
    bucketized = _assign_executor_actors(bucketized, sequence)

    return bucketized


def _assign_executor_actors(
    phases   : List[Dict[str, Any]],
    sequence : Dict[str, Any],
    approach_window: int = 10,
) -> List[Dict[str, Any]]:
    """
    为每个 phase 分配 executor_actor (P1/P2)。

    策略：
      1. is_split_child（均匀 bucketize 切出的子段）：按全局索引奇偶交替
      2. has_contact=True（语义接触段）：检查接触开始前 approach_window 帧内
         谁向对方靠近更多（投影位移更大）→ 该人为 executor
      3. has_contact=False（无接触段，如 walk_towards）：谁在该段内移动更多
         → 该人为 executor

    所有计算基于 XZ 平面（忽略 Y 轴高度差异）。
    """
    p1_roots = sequence["p1_root"]   # (T, 3)
    p2_roots = sequence["p2_root"]
    T_full   = len(p1_roots)

    def _xz(v):
        return v[[0, 2]]

    def _approaching_executor(t_s: int) -> str:
        """在 [t_look, t_s] 内谁向对方靠近更多。"""
        t_look = max(0, t_s - approach_window)
        if t_look >= t_s:
            return "P1"
        p1_s = _xz(p1_roots[t_look]);  p1_e = _xz(p1_roots[min(t_s, T_full - 1)])
        p2_s = _xz(p2_roots[t_look]);  p2_e = _xz(p2_roots[min(t_s, T_full - 1)])

        dir_p1_to_p2 = p2_s - p1_s
        norm = np.linalg.norm(dir_p1_to_p2)
        if norm < 1e-6:
            return "P1"
        dir_p1_to_p2 /= norm

        p1_toward = float(np.dot(p1_e - p1_s,  dir_p1_to_p2))
        p2_toward = float(np.dot(p2_e - p2_s, -dir_p1_to_p2))
        return "P1" if p1_toward >= p2_toward else "P2"

    def _moving_executor(t_s: int, t_e: int) -> str:
        """在 [t_s, t_e] 内谁移动更多（总位移长度）。"""
        t_e = min(t_e, T_full - 1)
        p1_disp = float(np.linalg.norm(_xz(p1_roots[t_e]) - _xz(p1_roots[t_s])))
        p2_disp = float(np.linalg.norm(_xz(p2_roots[t_e]) - _xz(p2_roots[t_s])))
        return "P1" if p1_disp >= p2_disp else "P2"

    for i, phase in enumerate(phases):
        if phase.get("is_split_child"):
            # 连续接触段均匀切分 → 交替
            phase["executor_actor"] = "P1" if i % 2 == 0 else "P2"
        elif phase.get("has_contact"):
            # 接触段 → 谁先靠近
            phase["executor_actor"] = _approaching_executor(phase["t_start"])
        else:
            # 无接触段 → 谁动得更多
            phase["executor_actor"] = _moving_executor(phase["t_start"], phase["t_end"])

    return phases


# ── LLM text annotation ───────────────────────────────────────────────────────

_PHASE_TYPE_LIST_ANNOTATOR = (
    '"approach", "contact", "release", "non_contact_interact"'
)

_ANNOTATOR_SYSTEM_PROMPT = """You are a motion description writer for two-person interaction data.
You receive structured motion facts measured from 3D joint positions.
Rewrite them as two optimized prompts — one for each person.

Output ONLY valid JSON (no markdown, no explanation):
{
  "p1_text": "Person 1 steps forward and extends their right arm toward Person 2, closing the gap.",
  "p2_text": "Person 2 stands still with arms open, waiting to receive Person 1's approach."
}

Rules:
1. Each sentence must be fluent, natural English — not a list of facts.
2. Use specific body parts and directions from the structured facts (e.g. "right arm", "toward").
3. Keep each sentence under 35 words.
4. Person 1 is always the subject in p1_text; Person 2 is always the subject in p2_text.
5. If the person is passive/stationary, describe their posture: "Person 2 stands still, arms at sides."
6. For contact phases, explicitly state whose body part contacts whose: e.g. "Person 1's right hand touches Person 2's left shoulder", "Person 2's chest meets Person 1's chest". Both sentences must name the contact pair.
7. NEVER write "Person 1 ... Person 1" or "Person 2 ... Person 2" — self-reference is always wrong.
8. NEVER include specific numbers or measurements (e.g. "0.80 m/s", "1.2m"). Use qualitative words instead: slow/steady/quickly, near/close/far.
9. NEVER use directional prepositions like "from behind", "from the front", "from the left/right", "from the side", "behind", "in front of", "to the left/right of" to describe WHERE one person is relative to the other — these are ambiguous. Instead describe each person's own body actions from their own perspective: "extends their right arm toward Person 2", "turns their torso", "steps forward", "raises both arms". For approach/separation, use: "moves closer", "steps toward Person 2", "moves away", "pulls back", "closes the gap", "increases distance".
"""


def _fix_self_reference(text: str, subject: str) -> str:
    """Fix LLM self-reference errors, e.g. 'Person 2 walks toward Person 2' → '...Person 1'.

    subject is "P1" or "P2" (internal label). The text uses "Person 1"/"Person 2".
    We fix occurrences of the subject label that appear after the first token.
    """
    import re
    subject_word = "Person 1" if subject == "P1" else "Person 2"
    other_word   = "Person 2" if subject == "P1" else "Person 1"
    if not text:
        return text
    # Ensure text uses "Person N" form (backward-compat with old P1/P2 cached texts)
    text = re.sub(r'\bP1\b', "Person 1", text)
    text = re.sub(r'\bP2\b', "Person 2", text)
    # Remove self-reference after the leading subject
    first, _, rest = text.partition(" ")
    if not rest:
        return text
    fixed_rest = re.sub(re.escape(subject_word), other_word, rest)
    return first + " " + fixed_rest


# SMPL-X joint indices (55-joint layout)
_WRIST_L_IDX = 20
_WRIST_R_IDX = 21


def _extract_motion_features_text(
    sequence : Dict[str, Any],
    t_start  : int,
    t_end    : int,
    fps      : int = 30,
) -> str:
    """保留旧接口，内部调用 _generate_structured_description。"""
    return _generate_structured_description(
        sequence, t_start, t_end, fps=fps,
    )


# SMPL 22-joint body indices
_J_PELVIS                 = 0
_J_L_HIP, _J_R_HIP       = 1, 2
_J_SPINE1, _J_SPINE2, _J_SPINE3 = 3, 6, 9
_J_L_KNEE, _J_R_KNEE     = 4, 5
_J_L_ANKLE, _J_R_ANKLE   = 7, 8
_J_L_SHOULDER, _J_R_SHOULDER = 16, 17
_J_L_ELBOW, _J_R_ELBOW   = 18, 19
_J_L_WRIST, _J_R_WRIST   = 20, 21
_J_NECK, _J_HEAD          = 12, 15
_CONTACT_JOINT_NAMES = {
    12: "neck", 15: "head",
    16: "left_shoulder", 17: "right_shoulder",
    18: "left_elbow",    19: "right_elbow",
    20: "left_wrist",    21: "right_wrist",
}
_UPPER_JOINTS = [12, 15, 16, 17, 18, 19]


def _facing_vector(joints: np.ndarray) -> np.ndarray:
    """(L, J, 3) → (L, 3) per-frame XZ forward vector (shoulder cross product)."""
    right = joints[:, _J_R_SHOULDER, :] - joints[:, _J_L_SHOULDER, :]  # (L, 3)
    right[:, 1] = 0.0
    norm = np.linalg.norm(right, axis=-1, keepdims=True) + 1e-8
    right = right / norm
    # forward = cross(right, up=(0,1,0)) → (-rz, 0, rx)
    fwd = np.stack([-right[:, 2], np.zeros(len(right)), right[:, 0]], axis=-1)
    return fwd


def _generate_structured_description(
    sequence    : Dict[str, Any],
    t_start     : int,
    t_end       : int,
    phase_type  : str = "",
    has_contact : bool = False,
    executor    : str = "P1",
    fps         : int = 30,
) -> str:
    """
    从物理信号生成结构化自然语言描述，供 LLM 改写为流畅英文。

    覆盖：根距离变化、双人移动速度与方向、朝向对齐度、
    手臂展开方向与高度、接触关节对。
    """
    p1_root   = sequence.get("p1_root")
    p2_root   = sequence.get("p2_root")
    p1_joints = sequence.get("p1_joints")
    p2_joints = sequence.get("p2_joints")

    if p1_root is None or p2_root is None:
        return ""

    t_start = max(0, t_start)
    t_end   = min(len(p1_root), t_end)
    if t_end <= t_start + 2:
        return ""

    p1r = p1_root[t_start:t_end]   # (L, 3)
    p2r = p2_root[t_start:t_end]
    L   = len(p1r)

    dist       = np.linalg.norm(p1r - p2r, axis=-1)
    dist_start = float(dist[0])
    dist_end   = float(dist[-1])
    dist_delta = dist_end - dist_start
    trend      = ("closing" if dist_delta < -0.05
                  else "opening" if dist_delta > 0.05 else "stable")

    # Minimum approach distance and timing
    min_dist     = float(dist.min())
    min_dist_idx = int(dist.argmin())
    if min_dist_idx < L // 3:
        min_timing = "early in phase"
    elif min_dist_idx < 2 * L // 3:
        min_timing = "mid-phase"
    else:
        min_timing = "late in phase"

    lines = [f"Distance: {dist_start:.2f}m → {dist_end:.2f}m ({trend}), closest {min_dist:.2f}m ({min_timing})"]

    # ── Who initiates movement ────────────────────────────────────────────────
    init_frames = min(10, L - 1)
    if init_frames > 0:
        p1_init_speed = float(np.linalg.norm(np.diff(p1r[:init_frames+1], axis=0), axis=-1).mean()) * fps
        p2_init_speed = float(np.linalg.norm(np.diff(p2r[:init_frames+1], axis=0), axis=-1).mean()) * fps
        if p1_init_speed > 0.15 and p2_init_speed <= 0.15:
            lines.append("Initiator: Person 1 initiates")
        elif p2_init_speed > 0.15 and p1_init_speed <= 0.15:
            lines.append("Initiator: Person 2 initiates")
        elif p1_init_speed > 0.15 and p2_init_speed > 0.15:
            lines.append("Initiator: both start moving simultaneously")
        else:
            lines.append("Initiator: both start stationary")

    # ── Per-person root movement ──────────────────────────────────────────────
    for pname, root, other_root in [
        ("Person 1", p1r, p2r),
        ("Person 2", p2r, p1r),
    ]:
        role      = "executor" if (pname == "Person 1") == (executor == "P1") else "passive"
        frame_vel = np.linalg.norm(np.diff(root, axis=0), axis=-1) * fps
        speed     = float(frame_vel.mean())
        moving    = speed > 0.15

        toward = other_root[0] - root[0]; toward[[1]] = 0.0
        tn   = np.linalg.norm(toward)
        disp = root[-1] - root[0]; disp[1] = 0.0
        dn   = np.linalg.norm(disp)
        if moving and tn > 0.1 and dn > 0.05:
            cos_a = float(np.dot(toward / tn, disp / dn))
            direction = ("toward the other person" if cos_a > 0.3
                         else "away from the other person" if cos_a < -0.3
                         else "sideways")
            lines.append(f"{pname} ({role}): moving {direction}")
        elif moving:
            lines.append(f"{pname} ({role}): moving")
        else:
            lines.append(f"{pname} ({role}): stationary")

    if p1_joints is None or p2_joints is None:
        return "\n".join(lines)

    p1j = p1_joints[t_start:t_end]   # (L, J, 3)
    p2j = p2_joints[t_start:t_end]

    # ── Facing alignment ─────────────────────────────────────────────────────
    p1_fwd = _facing_vector(p1j)   # (L, 3)
    p2_fwd = _facing_vector(p2j)
    # face-to-face: p1_fwd antiparallel to p2_fwd
    align  = float(np.mean(np.sum(p1_fwd * (-p2_fwd), axis=-1)))
    if align > 0.7:
        facing_str = f"face-to-face ({align:.2f})"
    elif align > 0.3:
        facing_str = f"partially facing ({align:.2f})"
    else:
        facing_str = f"not facing each other ({align:.2f})"
    lines.append(f"Facing: {facing_str}")

    # ── Single-person facing toward other ────────────────────────────────────
    for pname, fwd, my_root, other_root in [
        ("Person 1", p1_fwd, p1r, p2r),
        ("Person 2", p2_fwd, p2r, p1r),
    ]:
        to_other = other_root - my_root   # (L, 3)
        to_other[:, 1] = 0.0
        to_other_n = to_other / (np.linalg.norm(to_other, axis=-1, keepdims=True) + 1e-8)
        cos_face = float(np.mean(np.sum(fwd * to_other_n, axis=-1)))
        if cos_face > 0.7:
            face_str = "facing toward the other"
        elif cos_face > 0.2:
            face_str = "partially facing the other"
        else:
            face_str = "facing away from the other"
        lines.append(f"{pname} gaze: {face_str}")

    # ── Relative position (each person's body frame) ─────────────────────────
    p1_fwd_m = p1_fwd.mean(axis=0); p1_fwd_m[1] = 0.0
    p1_fwd_n  = p1_fwd_m / (np.linalg.norm(p1_fwd_m) + 1e-8)
    p1_right_n = np.array([p1_fwd_n[2], 0.0, -p1_fwd_n[0]])
    p2_fwd_m = p2_fwd.mean(axis=0); p2_fwd_m[1] = 0.0
    p2_fwd_n  = p2_fwd_m / (np.linalg.norm(p2_fwd_m) + 1e-8)
    p2_right_n = np.array([p2_fwd_n[2], 0.0, -p2_fwd_n[0]])

    for pname, my_fwd_n, my_right_n, my_r, other_r, other_name in [
        ("Person 1", p1_fwd_n, p1_right_n, p1r, p2r, "Person 2"),
        ("Person 2", p2_fwd_n, p2_right_n, p2r, p1r, "Person 1"),
    ]:
        rel_xz = other_r.mean(axis=0) - my_r.mean(axis=0); rel_xz[1] = 0.0
        fwd_proj   = float(np.dot(rel_xz, my_fwd_n))
        right_proj = float(np.dot(rel_xz, my_right_n))
        if fwd_proj > 0.2:
            rel_pos = f"{other_name} is in front"
        elif fwd_proj < -0.2:
            rel_pos = f"{other_name} is behind"
        else:
            rel_pos = f"{other_name} is to the side"
        if abs(right_proj) > 0.3:
            rel_pos += " (right)" if right_proj > 0 else " (left)"
        lines.append(f"{pname} view: {rel_pos}")

    # ── Approach angle (per person: where is the other relative to my facing) ──
    for pname, my_fwd_n, rel_to_me in [
        ("Person 1", p1_fwd_n, p2r.mean(axis=0) - p1r.mean(axis=0)),
        ("Person 2", p2_fwd_n, p1r.mean(axis=0) - p2r.mean(axis=0)),
    ]:
        d = rel_to_me.copy(); d[1] = 0.0
        dn = np.linalg.norm(d) + 1e-8
        cos_a = float(np.clip(np.dot(d / dn, my_fwd_n), -1.0, 1.0))
        ang = float(np.degrees(np.arccos(cos_a)))
        if ang < 30:
            astr = "other person is directly in front"
        elif ang < 70:
            astr = "other person is diagonally ahead"
        elif ang < 110:
            astr = "other person is to the side"
        elif ang < 150:
            astr = "other person is diagonally behind"
        else:
            astr = "other person is directly behind"
        lines.append(f"{pname} relative view: {astr}")

    # ── Body yaw change per person ────────────────────────────────────────────
    for pname, pj in [("Person 1", p1j), ("Person 2", p2j)]:
        fwd = _facing_vector(pj)   # (L, 3)
        cos_turn = float(np.clip(np.dot(fwd[0], fwd[-1]), -1.0, 1.0))
        turn_deg = float(np.degrees(np.arccos(cos_turn)))
        cross_y  = float(fwd[0, 0] * fwd[-1, 2] - fwd[0, 2] * fwd[-1, 0])
        if turn_deg < 15:
            yaw_str = "no significant turn"
        elif turn_deg < 45:
            direction_t = "right" if cross_y < 0 else "left"
            yaw_str = f"slight {direction_t} turn"
        elif turn_deg < 90:
            direction_t = "right" if cross_y < 0 else "left"
            yaw_str = f"turning {direction_t}"
        else:
            direction_t = "right" if cross_y < 0 else "left"
            yaw_str = f"large {direction_t} turn"
        lines.append(f"{pname} body turn: {yaw_str}")

    # ── Mutual reach symmetry ─────────────────────────────────────────────────
    def _min_wrist_to_upper(pj_src, pj_tgt):
        wrists = np.stack([pj_src[:, _J_L_WRIST, :], pj_src[:, _J_R_WRIST, :]], axis=1)  # (L,2,3)
        upper  = pj_tgt[:, _UPPER_JOINTS, :]   # (L,6,3)
        dists  = np.linalg.norm(wrists[:, :, None, :] - upper[:, None, :, :], axis=-1)  # (L,2,6)
        return float(dists.min(axis=(1, 2)).mean())

    d_p1_to_p2 = _min_wrist_to_upper(p1j, p2j)
    d_p2_to_p1 = _min_wrist_to_upper(p2j, p1j)
    both_reaching   = d_p1_to_p2 < 0.40 and d_p2_to_p1 < 0.40
    only_p1_reaches = d_p1_to_p2 < 0.40 and d_p2_to_p1 >= 0.40
    only_p2_reaches = d_p2_to_p1 < 0.40 and d_p1_to_p2 >= 0.40
    if both_reaching:
        reach_str = "both reaching toward each other"
    elif only_p1_reaches:
        reach_str = "Person 1 reaching toward Person 2"
    elif only_p2_reaches:
        reach_str = "Person 2 reaching toward Person 1"
    else:
        reach_str = "neither reaching"
    lines.append(f"Arm reach: {reach_str}")

    # ── Foot stepping per person ──────────────────────────────────────────────
    for pname, pj in [("Person 1", p1j), ("Person 2", p2j)]:
        l_ankle_y = pj[:, _J_L_ANKLE, 1]
        r_ankle_y = pj[:, _J_R_ANKLE, 1]
        l_lift = float(l_ankle_y.max() - l_ankle_y.min())
        r_lift = float(r_ankle_y.max() - r_ankle_y.min())
        max_lift = max(l_lift, r_lift)
        if max_lift > 0.15:
            step_str = "taking steps"
        elif max_lift > 0.05:
            step_str = "small foot adjustments"
        else:
            step_str = "feet planted"
        lines.append(f"{pname} feet: {step_str}")

    # ── Pelvis height difference ──────────────────────────────────────────────
    p1_pelvis_h = float(p1j[:, _J_PELVIS, 1].mean())
    p2_pelvis_h = float(p2j[:, _J_PELVIS, 1].mean())
    h_diff = p1_pelvis_h - p2_pelvis_h
    if abs(h_diff) > 0.12:
        lower = "Person 1" if h_diff < 0 else "Person 2"
        lines.append(f"Height: {lower} is lower (crouching or height difference)")
    else:
        lines.append("Height: similar standing height")

    # ── Elbow bend angle per person ───────────────────────────────────────────
    for pname, pj in [("Person 1", p1j), ("Person 2", p2j)]:
        elbow_parts = []
        for side, j_sh, j_el, j_wr in [
            ("right", _J_R_SHOULDER, _J_R_ELBOW, _J_R_WRIST),
            ("left",  _J_L_SHOULDER, _J_L_ELBOW, _J_L_WRIST),
        ]:
            upper = pj[:, j_el, :] - pj[:, j_sh, :]   # (L, 3)
            lower = pj[:, j_wr, :] - pj[:, j_el, :]
            upper_n = upper / (np.linalg.norm(upper, axis=-1, keepdims=True) + 1e-8)
            lower_n = lower / (np.linalg.norm(lower, axis=-1, keepdims=True) + 1e-8)
            cos_angle = float(np.mean(np.sum(upper_n * lower_n, axis=-1)))
            # cos_angle ≈ 1 → straight arm, cos_angle ≈ -1 → fully bent
            if cos_angle > 0.7:
                bend = "extended"
            elif cos_angle > 0.0:
                bend = "slightly bent"
            else:
                bend = "bent"
            elbow_parts.append(f"{side} arm {bend}")
        lines.append(f"{pname} elbows: " + ", ".join(elbow_parts))

    # ── Knee bend per person ──────────────────────────────────────────────────
    for pname, pj in [("Person 1", p1j), ("Person 2", p2j)]:
        knee_parts = []
        for side, j_hip, j_kn, j_ank in [
            ("right", _J_R_HIP, _J_R_KNEE, _J_R_ANKLE),
            ("left",  _J_L_HIP, _J_L_KNEE, _J_L_ANKLE),
        ]:
            upper = pj[:, j_kn, :] - pj[:, j_hip, :]
            lower = pj[:, j_ank, :] - pj[:, j_kn, :]
            upper_n = upper / (np.linalg.norm(upper, axis=-1, keepdims=True) + 1e-8)
            lower_n = lower / (np.linalg.norm(lower, axis=-1, keepdims=True) + 1e-8)
            cos_k = float(np.mean(np.sum(upper_n * lower_n, axis=-1)))
            if cos_k > 0.85:
                knee_parts.append(f"{side} leg straight")
            elif cos_k > 0.5:
                knee_parts.append(f"{side} knee slightly bent")
            else:
                knee_parts.append(f"{side} knee bent")
        lines.append(f"{pname} legs: " + ", ".join(knee_parts))

    # ── Torso posture per person ──────────────────────────────────────────────
    for pname, pj, pj_fwd in [("Person 1", p1j, p1_fwd), ("Person 2", p2j, p2_fwd)]:
        pelvis  = pj[:, _J_PELVIS, :]    # (L, 3)
        spine3  = pj[:, _J_SPINE3, :]
        neck    = pj[:, _J_NECK,   :]
        head    = pj[:, _J_HEAD,   :]
        l_sh    = pj[:, _J_L_SHOULDER, :]
        r_sh    = pj[:, _J_R_SHOULDER, :]

        # Spine lean: forward tilt + direction (toward/away from other)
        trunk_vec = spine3 - pelvis   # (L, 3)
        trunk_xz  = np.stack([trunk_vec[:, 0], trunk_vec[:, 2]], axis=-1)
        trunk_y   = trunk_vec[:, 1]
        lean      = float(np.mean(np.linalg.norm(trunk_xz, axis=-1) / (np.abs(trunk_y) + 1e-6)))
        other_r   = p2r if pname == "Person 1" else p1r
        to_other  = other_r - (p1r if pname == "Person 1" else p2r)
        to_other[:, 1] = 0.0
        to_other_n = to_other / (np.linalg.norm(to_other, axis=-1, keepdims=True) + 1e-8)
        trunk_xz_n = trunk_xz / (np.linalg.norm(trunk_xz, axis=-1, keepdims=True) + 1e-8)
        lean_toward = float(np.mean(np.sum(trunk_xz_n * to_other_n[:, [0, 2]], axis=-1)))
        if lean > 0.25:
            lean_dir = "toward the other" if lean_toward > 0.3 else \
                       "away from the other" if lean_toward < -0.3 else "sideways"
            lean_str = f"leaning forward noticeably ({lean_dir})"
        elif lean > 0.10:
            lean_dir = "toward the other" if lean_toward > 0.3 else \
                       "away from the other" if lean_toward < -0.3 else "sideways"
            lean_str = f"leaning slightly ({lean_dir})"
        else:
            lean_str = "upright"

        # Head pitch: forward projection along facing direction
        head_vec = head - neck
        head_fwd_proj = float(np.mean(
            head_vec[:, 0] * pj_fwd[:, 0] + head_vec[:, 2] * pj_fwd[:, 2]
        ))
        head_y = float(np.mean(head_vec[:, 1]))
        if head_fwd_proj > 0.04:
            head_str = "head pitched forward (looking down)"
        elif head_y > 0.06 or head_fwd_proj < -0.03:
            head_str = "head raised (looking up)"
        else:
            head_str = "head upright"

        # Lateral tilt: shoulder height asymmetry
        sh_diff = float(np.mean(r_sh[:, 1] - l_sh[:, 1]))
        if sh_diff > 0.06:
            tilt_str = "tilted right"
        elif sh_diff < -0.06:
            tilt_str = "tilted left"
        else:
            tilt_str = None

        posture_parts = [lean_str, head_str]
        if tilt_str:
            posture_parts.append(tilt_str)
        lines.append(f"{pname} torso: " + ", ".join(posture_parts))

    # ── Weight shift / lateral balance ───────────────────────────────────────
    for pname, pj, pj_fwd in [("Person 1", p1j, p1_fwd), ("Person 2", p2j, p2_fwd)]:
        pelvis_xz   = pj[:, _J_PELVIS, [0, 2]]
        l_ank_xz    = pj[:, _J_L_ANKLE, [0, 2]]
        r_ank_xz    = pj[:, _J_R_ANKLE, [0, 2]]
        foot_mid_xz = (l_ank_xz + r_ank_xz) / 2
        # right lateral axis: forward rotated 90° CW in XZ → (fz, -fx)
        right_xz = np.stack([pj_fwd[:, 2], -pj_fwd[:, 0]], axis=-1)
        lateral_shift = float(np.mean(
            np.sum((pelvis_xz - foot_mid_xz) * right_xz, axis=-1)
        ))
        if lateral_shift > 0.06:
            shift_str = "weight shifted to right side"
        elif lateral_shift < -0.06:
            shift_str = "weight shifted to left side"
        else:
            shift_str = "weight evenly distributed"
        lines.append(f"{pname} balance: {shift_str}")

    # ── Arm extension per person ──────────────────────────────────────────────
    for pname, pj, oj in [("Person 1", p1j, p2j), ("Person 2", p2j, p1j)]:
        hip_h    = float(np.mean((pj[:, _J_L_HIP, 1] + pj[:, _J_R_HIP, 1]) / 2))
        sh_h     = float(np.mean((pj[:, _J_L_SHOULDER, 1] + pj[:, _J_R_SHOULDER, 1]) / 2))
        le_h     = float(np.mean(pj[:, _J_L_ELBOW, 1]))
        re_h     = float(np.mean(pj[:, _J_R_ELBOW, 1]))
        lw_h     = float(np.mean(pj[:, _J_L_WRIST, 1]))
        rw_h     = float(np.mean(pj[:, _J_R_WRIST, 1]))
        lw_pos   = pj[:, _J_L_WRIST, :]
        rw_pos   = pj[:, _J_R_WRIST, :]
        le_pos   = pj[:, _J_L_ELBOW, :]
        re_pos   = pj[:, _J_R_ELBOW, :]
        other_upper = oj[:, _UPPER_JOINTS, :]

        def _min_dist_to_upper(pos):
            return float(np.mean(
                np.linalg.norm(other_upper - pos[:, None, :], axis=-1).min(axis=-1)
            ))

        lw_dist = _min_dist_to_upper(lw_pos)
        rw_dist = _min_dist_to_upper(rw_pos)
        le_dist = _min_dist_to_upper(le_pos)
        re_dist = _min_dist_to_upper(re_pos)

        arm_parts = []
        # Wrist height relative to hip (raised arm)
        if rw_h > sh_h - 0.05:
            arm_parts.append("right arm fully raised")
        elif rw_h > hip_h + 0.35:
            arm_parts.append("right arm raised")
        if lw_h > sh_h - 0.05:
            arm_parts.append("left arm fully raised")
        elif lw_h > hip_h + 0.35:
            arm_parts.append("left arm raised")
        # Elbow proximity to other person
        for side, ed in [("right elbow", re_dist), ("left elbow", le_dist)]:
            if ed < 0.30:
                arm_parts.append(f"{side} close to other person")
        # Wrist proximity to other person
        min_wd = min(lw_dist, rw_dist)
        if min_wd < 0.35:
            side = "right" if rw_dist < lw_dist else "left"
            arm_parts.append(f"{side} wrist near other person")
        # Wrist crossing body midline (reaching across to other side)
        torso_cx = float(np.mean((pj[:, _J_L_SHOULDER, 0] + pj[:, _J_R_SHOULDER, 0]) / 2))
        lw_cx = float(np.mean(lw_pos[:, 0]))
        rw_cx = float(np.mean(rw_pos[:, 0]))
        if lw_cx > torso_cx + 0.08:
            arm_parts.append("left wrist crossing midline")
        if rw_cx < torso_cx - 0.08:
            arm_parts.append("right wrist crossing midline")
        if arm_parts:
            lines.append(f"{pname} arms: " + "; ".join(arm_parts))

    # ── Arm spread angle per person (hug indicator) ───────────────────────────
    for pname, pj in [("Person 1", p1j), ("Person 2", p2j)]:
        lw = pj[:, _J_L_WRIST, :] - pj[:, _J_L_SHOULDER, :]
        rw = pj[:, _J_R_WRIST, :] - pj[:, _J_R_SHOULDER, :]
        lw_xz = lw[:, [0, 2]]; rw_xz = rw[:, [0, 2]]
        ln = np.linalg.norm(lw_xz, axis=-1, keepdims=True) + 1e-8
        rn = np.linalg.norm(rw_xz, axis=-1, keepdims=True) + 1e-8
        cos_spread = float(np.mean(np.sum((lw_xz / ln) * (-rw_xz / rn), axis=-1)))
        if cos_spread > 0.5:
            spread_str = "arms spread wide open"
        elif cos_spread > 0.0:
            spread_str = "arms moderately open"
        else:
            spread_str = "arms close to body"
        lines.append(f"{pname} arm spread: {spread_str}")

    # ── Arm movement direction per person ────────────────────────────────────
    for pname, pj, pj_fwd in [("Person 1", p1j, p1_fwd), ("Person 2", p2j, p2_fwd)]:
        arm_dir_parts = []
        for side, j_sh, j_wr in [
            ("right", _J_R_SHOULDER, _J_R_WRIST),
            ("left",  _J_L_SHOULDER, _J_L_WRIST),
        ]:
            arm_vec = pj[:, j_wr, :] - pj[:, j_sh, :]
            arm_fwd = float(np.mean(arm_vec[:, 0] * pj_fwd[:, 0] + arm_vec[:, 2] * pj_fwd[:, 2]))
            arm_up  = float(np.mean(arm_vec[:, 1]))
            if arm_up > 0.25 and arm_up > abs(arm_fwd):
                arm_dir_parts.append(f"{side} arm reaching upward")
            elif arm_fwd > 0.20:
                arm_dir_parts.append(f"{side} arm reaching forward")
            elif arm_fwd < -0.10:
                arm_dir_parts.append(f"{side} arm extended backward (behind torso)")
            else:
                arm_dir_parts.append(f"{side} arm at side")
        lines.append(f"{pname} arm direction: " + ", ".join(arm_dir_parts))

    # ── Bilateral arm symmetry ────────────────────────────────────────────────
    for pname, pj, pj_fwd in [("Person 1", p1j, p1_fwd), ("Person 2", p2j, p2_fwd)]:
        l_vec = pj[:, _J_L_WRIST, :] - pj[:, _J_L_SHOULDER, :]
        r_vec = pj[:, _J_R_WRIST, :] - pj[:, _J_R_SHOULDER, :]
        l_fwd = float(np.mean(l_vec[:, 0] * pj_fwd[:, 0] + l_vec[:, 2] * pj_fwd[:, 2]))
        r_fwd = float(np.mean(r_vec[:, 0] * pj_fwd[:, 0] + r_vec[:, 2] * pj_fwd[:, 2]))
        l_up  = float(np.mean(l_vec[:, 1]))
        r_up  = float(np.mean(r_vec[:, 1]))
        height_asym = abs(l_up - r_up)
        reach_asym  = abs(l_fwd - r_fwd)
        if height_asym < 0.10 and reach_asym < 0.15:
            sym_str = "arms symmetric"
        else:
            details = []
            if height_asym >= 0.10:
                details.append("height")
            if reach_asym >= 0.15:
                details.append("reach")
            sym_str = "arms asymmetric (" + "/".join(details) + ")"
        lines.append(f"{pname} arm symmetry: {sym_str}")

    # ── Spine axial twist ─────────────────────────────────────────────────────
    for pname, pj in [("Person 1", p1j), ("Person 2", p2j)]:
        pelvis_right = pj[:, _J_R_HIP, :] - pj[:, _J_L_HIP, :]
        spine_right  = pj[:, _J_R_SHOULDER, :] - pj[:, _J_L_SHOULDER, :]
        # project to XZ, measure rotation between hip axis and shoulder axis
        pr_xz = pelvis_right[:, [0, 2]]
        sr_xz = spine_right[:,  [0, 2]]
        pr_n  = pr_xz / (np.linalg.norm(pr_xz, axis=-1, keepdims=True) + 1e-8)
        sr_n  = sr_xz / (np.linalg.norm(sr_xz, axis=-1, keepdims=True) + 1e-8)
        cos_twist = float(np.mean(np.sum(pr_n * sr_n, axis=-1)))
        twist_deg = float(np.degrees(np.arccos(np.clip(cos_twist, -1.0, 1.0))))
        if twist_deg > 30:
            twist_str = "upper body rotated significantly"
        elif twist_deg > 15:
            twist_str = "slight upper body rotation"
        else:
            twist_str = "no upper body twist"
        lines.append(f"{pname} spine twist: {twist_str}")

    # ── Movement synchrony ────────────────────────────────────────────────────
    p1_disp = p1r[-1] - p1r[0]; p1_disp[1] = 0.0
    p2_disp = p2r[-1] - p2r[0]; p2_disp[1] = 0.0
    p1_dn = np.linalg.norm(p1_disp) + 1e-8
    p2_dn = np.linalg.norm(p2_disp) + 1e-8
    p1_moving = float(np.linalg.norm(np.diff(p1r, axis=0), axis=-1).mean()) * fps > 0.15
    p2_moving = float(np.linalg.norm(np.diff(p2r, axis=0), axis=-1).mean()) * fps > 0.15
    if p1_moving and p2_moving:
        cos_sync = float(np.dot(p1_disp / p1_dn, p2_disp / p2_dn))
        if cos_sync > 0.7:
            sync_str = "both moving in the same direction"
        elif cos_sync < -0.3:
            sync_str = "moving toward/away from each other simultaneously"
        else:
            sync_str = "moving in different directions"
    elif p1_moving and not p2_moving:
        sync_str = "only Person 1 moving"
    elif p2_moving and not p1_moving:
        sync_str = "only Person 2 moving"
    else:
        sync_str = "both stationary"
    lines.append(f"Movement sync: {sync_str}")

    # ── Movement axis: lateral vs forward per person ──────────────────────────
    for pname, pr, pj_fwd in [("Person 1", p1r, p1_fwd), ("Person 2", p2r, p2_fwd)]:
        disp = pr[-1] - pr[0]; disp[1] = 0.0
        disp_mag = float(np.linalg.norm(disp))
        if disp_mag > 0.05:
            fwd_m = pj_fwd.mean(axis=0); fwd_m[1] = 0.0
            fwd_m /= (np.linalg.norm(fwd_m) + 1e-8)
            right_m = np.array([fwd_m[2], 0.0, -fwd_m[0]])
            fwd_comp = abs(float(np.dot(disp, fwd_m)))
            lat_comp = abs(float(np.dot(disp, right_m)))
            ratio = lat_comp / (fwd_comp + lat_comp + 1e-8)
            if ratio > 0.6:
                axis_str = "primarily lateral (sidestep)"
            elif ratio > 0.35:
                axis_str = "diagonal (forward + lateral)"
            else:
                axis_str = "primarily forward/backward"
            lines.append(f"{pname} movement axis: {axis_str}")

    # ── Contact joint pair + duration ratio ───────────────────────────────────
    CONTACT_THRESH = 0.35
    cj_list = list(_CONTACT_JOINT_NAMES.keys())
    # Per-frame minimum joint distance
    per_frame_min = np.array([
        min(float(np.linalg.norm(p1j[t, j1, :] - p2j[t, j2, :]))
            for j1 in cj_list for j2 in cj_list)
        for t in range(len(p1j))
    ])
    contact_ratio = float((per_frame_min < CONTACT_THRESH).mean())
    if has_contact:
        if contact_ratio > 0.8:
            dur_str = "sustained throughout"
        elif contact_ratio > 0.4:
            dur_str = "intermittent"
        else:
            dur_str = "brief"
        # Relative motion during contact (static hold vs dynamic)
        contact_frames = per_frame_min < CONTACT_THRESH
        if contact_frames.sum() > 1:
            cf_idx = np.where(contact_frames)[0]
            p1_cf_vel = float(np.linalg.norm(np.diff(p1r[cf_idx], axis=0), axis=-1).mean()) * fps
            p2_cf_vel = float(np.linalg.norm(np.diff(p2r[cf_idx], axis=0), axis=-1).mean()) * fps
            avg_cf_vel = (p1_cf_vel + p2_cf_vel) / 2
            if avg_cf_vel < 0.10:
                contact_motion = "both nearly still during contact (static hold)"
            elif avg_cf_vel < 0.40:
                contact_motion = "gentle movement during contact"
            else:
                contact_motion = "active movement during contact (dynamic)"
            lines.append(f"Contact motion: {contact_motion}")

        # Best contact joint pair (mean distance over contact frames)
        if contact_frames.any():
            best_dist, best_j1, best_j2 = float("inf"), -1, -1
            for j1 in cj_list:
                for j2 in cj_list:
                    d = float(np.mean(np.linalg.norm(
                        p1j[contact_frames, j1, :] - p2j[contact_frames, j2, :], axis=-1)))
                    if d < best_dist:
                        best_dist, best_j1, best_j2 = d, j1, j2
            # Count simultaneous contact points
            multi_thresh = CONTACT_THRESH * 1.2
            n_contact_pairs = sum(
                1 for j1 in cj_list for j2 in cj_list
                if float(np.mean(np.linalg.norm(
                    p1j[contact_frames, j1, :] - p2j[contact_frames, j2, :], axis=-1))) < multi_thresh
            )
            n_contact_pts = min(n_contact_pairs, 5)
            multi_str = ("single point" if n_contact_pts <= 1
                         else "two contact points" if n_contact_pts <= 3
                         else "multiple contact points")
            if best_j1 >= 0:
                n1 = _CONTACT_JOINT_NAMES[best_j1]
                n2 = _CONTACT_JOINT_NAMES[best_j2]
                lines.append(
                    f"Contact: Person 1 {n1} ↔ Person 2 {n2}, "
                    f"{dur_str}, {multi_str}"
                )
                # Contact body region
                _HAND_J  = {_J_L_WRIST, _J_R_WRIST}
                _ARM_J   = {_J_L_ELBOW, _J_R_ELBOW, _J_L_WRIST, _J_R_WRIST}
                _SHLD_J  = {_J_L_SHOULDER, _J_R_SHOULDER}
                _HEAD_J  = {_J_NECK, _J_HEAD}
                j1s, j2s = {best_j1}, {best_j2}
                if j1s & _HAND_J and j2s & _HAND_J:
                    region = "hand-to-hand contact"
                elif j1s & _ARM_J and j2s & _ARM_J:
                    region = "arm-to-arm contact"
                elif (j1s & _ARM_J and j2s & _SHLD_J) or (j1s & _SHLD_J and j2s & _ARM_J):
                    region = "hand/arm to shoulder contact"
                elif j1s & _SHLD_J or j2s & _SHLD_J:
                    region = "shoulder contact"
                elif j1s & _HEAD_J or j2s & _HEAD_J:
                    region = "head/neck contact"
                else:
                    region = "torso/body contact"
                lines.append(f"Contact region: {region}")
                # Contact approach direction: executor position relative to passive body frame at contact
                first_cf = int(np.argmax(contact_frames))
                exec_r_cf = (p1r if executor == "P1" else p2r)[first_cf]
                pass_r_cf = (p2r if executor == "P1" else p1r)[first_cf]
                pass_fwd_cf = (p2_fwd if executor == "P1" else p1_fwd)[first_cf]
                rel_pos = exec_r_cf - pass_r_cf
                rel_pos[1] = 0.0
                rel_mag = float(np.linalg.norm(rel_pos))
                if rel_mag > 0.05:
                    rel_n = rel_pos / rel_mag
                    pass_fwd_xz = np.array([pass_fwd_cf[0], 0.0, pass_fwd_cf[2]])
                    pass_fwd_xz /= (np.linalg.norm(pass_fwd_xz) + 1e-8)
                    cos_ca = float(np.dot(rel_n, pass_fwd_xz))
                    if cos_ca > 0.5:
                        contact_dir = "face-to-face (both facing each other)"
                    elif cos_ca < -0.5:
                        contact_dir = "same-direction (one behind the other)"
                    else:
                        contact_dir = "side-by-side or perpendicular"
                    lines.append(f"Contact approach direction: {contact_dir}")

    # ── Final relative position (per person view) ────────────────────────────
    for pname, my_fwd_end, my_r_end, other_r_end, other_name in [
        ("Person 1", p1_fwd[-1], p1r[-1], p2r[-1], "Person 2"),
        ("Person 2", p2_fwd[-1], p2r[-1], p1r[-1], "Person 1"),
    ]:
        rel = other_r_end - my_r_end; rel[1] = 0.0
        fwd_e = my_fwd_end.copy(); fwd_e[1] = 0.0
        fwd_e /= (np.linalg.norm(fwd_e) + 1e-8)
        right_e = np.array([fwd_e[2], 0.0, -fwd_e[0]])
        fp = float(np.dot(rel, fwd_e))
        rp = float(np.dot(rel, right_e))
        if abs(fp) > abs(rp):
            end_pos = f"{other_name} is in front" if fp > 0 else f"{other_name} is behind"
        else:
            end_pos = f"{other_name} is to the {'right' if rp > 0 else 'left'}"
        lines.append(f"{pname} end view: {end_pos}")

    return "\n".join(lines)


def annotate_phases_with_llm(
    phases          : "List[Dict[str, Any]]",
    sequence        : "Dict[str, Any]",
    llm_model       : str = "qwen3.5:35b-a3b-bf16",
    llm_base_url    : "Optional[str]" = None,
    llm_cache_path  : "Optional[str]" = None,
    no_motion_facts : bool = False,
) -> "List[Dict[str, Any]]":
    """
    为每个 phase 调用 LLM 生成 P1/P2 的自然语言动作描述，写入 phase_meta：
      phase_meta["p1_text"] : str  — P1 的自然语言描述
      phase_meta["p2_text"] : str  — P2 的自然语言描述

    结果按 seq_id 缓存到 {llm_cache_path}/{seq_id}_texts.json，避免重复调用。
    """
    import json as _json
    import re as _re

    seq_id      = sequence.get("seq_id", "")
    global_text = sequence.get("global_text", "")

    action_label_match = _re.search(r"A(\d+)", seq_id)
    action_label = f"A{action_label_match.group(1)}" if action_label_match else "unknown"

    # ── 读取缓存 ────────────────────────────────────────────────────────────
    # 格式：{ seq_id: { "phase000": { phase_type, frame_range, p1_action, p2_action } } }
    cache: dict = {}       # phase_key → entry，从文件加载后 unwrap seq_id 层
    cache_file  = None

    def _is_lazy_annot(entry: dict) -> bool:
        """Return True if this cache entry contains fallback/lazy placeholder text."""
        _LAZY = {
            "P1 moves.", "P2 moves.",
            "P1 makes physical contact with P2.",
            "P2 makes physical contact with P1.",
            "Person 1 moves.", "Person 2 moves.",
            "Person 1 makes physical contact with Person 2.",
            "Person 2 makes physical contact with Person 1.",
        }
        p1 = entry.get("p1_action", "").strip()
        p2 = entry.get("p2_action", "").strip()
        return (not p1 or p1 in _LAZY) and (not p2 or p2 in _LAZY)

    if llm_cache_path:
        os.makedirs(llm_cache_path, exist_ok=True)
        cache_file = os.path.join(llm_cache_path, f"{seq_id}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file) as cf:
                    _raw = _json.load(cf)
                # cache files wrap phases under "local_phases" key
                _all = _raw.get("local_phases", _raw)
                # Drop lazy/fallback entries — they were never properly annotated
                # and must be re-queried on the next annotate run.
                cache = {k: v for k, v in _all.items() if not _is_lazy_annot(v)}
            except Exception:
                cache = {}

    resolved_base_url = llm_base_url or os.environ.get(
        "LLM_BASE_URL", "http://localhost:11435/v1"
    )
    resolved_api_key  = os.environ.get("OPENAI_API_KEY", "ollama")
    _is_ollama = "localhost" in resolved_base_url or "127.0.0.1" in resolved_base_url

    def _parse_json(raw: str) -> "Optional[dict]":
        raw = _re.sub(r"```(?:json)?\s*", "", raw)
        raw = _re.sub(r"```", "", raw).strip()
        try:
            return _json.loads(raw)
        except Exception:
            m = _re.search(r"\{.*\}", raw, _re.DOTALL)
            if m:
                try:
                    return _json.loads(m.group())
                except Exception:
                    pass
        return None

    def _send_messages(messages: list) -> "Optional[str]":
        """Send full messages list to LLM, return raw response string."""
        import urllib.request as _urlreq
        if _is_ollama:
            port_match = _re.search(r":(\d+)", resolved_base_url)
            port = port_match.group(1) if port_match else "11434"
            payload = _json.dumps({
                "model"   : llm_model,
                "think"   : False,
                "stream"  : False,
                "messages": messages,
                "options" : {"temperature": 0.1, "num_predict": 256},
            }).encode()
            try:
                req = _urlreq.Request(
                    f"http://localhost:{port}/api/chat",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with _urlreq.urlopen(req, timeout=120) as r:
                    return _json.loads(r.read()).get("message", {}).get("content", "") or ""
            except Exception as e:
                logger.error("Ollama API call failed: %s", e)
                return None
        else:
            try:
                import openai
                client = openai.OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
                resp = client.chat.completions.create(
                    model=llm_model, messages=messages,
                    max_tokens=256, temperature=0.1,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                return None

    # ── 初始化多轮 session（序列级 context 作为首条 user 消息）────────────────
    # system prompt + 序列 context 只发送一次，后续 phase 追加到同一 messages 列表，
    # 让 LLM server 的 KV-cache 复用已编码的 prefix。
    session_messages: list = [
        {"role": "system", "content": _ANNOTATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Sequence: {seq_id}\n"
                f"Action category: {action_label}\n"
                f"Global description: {global_text}\n\n"
                f"I will describe each phase one by one. "
                f"For each phase reply with a JSON object containing only "
                f"'p1_text' and 'p2_text'."
            ),
        },
        {"role": "assistant", "content": "Understood. Please describe each phase."},
    ]

    # 预计算：第一个接触段之前的最后一个无接触 phase 是 reach 候选
    first_contact_ids: set = set()
    _contact_seen = False
    _last_no_contact_id = None
    for ph in phases:
        if ph.get("has_contact", False) and not _contact_seen:
            _contact_seen = True
            if _last_no_contact_id:
                first_contact_ids.add(_last_no_contact_id)
        elif not ph.get("has_contact", False):
            _last_no_contact_id = ph.get("phase_id", "")

    for phase in phases:
        phase_id    = phase.get("phase_id", "")
        # cache key = short phase key, e.g. "phase000" (strip "{seq_id}_" prefix)
        phase_key   = phase_id[len(seq_id) + 1:] if phase_id.startswith(seq_id + "_") else phase_id
        has_contact = phase.get("has_contact", False)
        executor    = phase.get("executor_actor", "P1")
        passive     = "P2" if executor == "P1" else "P1"
        t_start     = phase.get("t_start", 0)
        t_end       = phase.get("t_end", 0)
        is_first_contact = phase_id in first_contact_ids

        # root 间距变化量（用于 step_back 修正）
        p1r = sequence.get("p1_root")
        p2r = sequence.get("p2_root")
        dist_delta: Optional[float] = None
        if p1r is not None and p2r is not None and t_end > t_start:
            import numpy as _np
            d = _np.linalg.norm(p1r[t_start:t_end] - p2r[t_start:t_end], axis=-1)
            dist_delta = float(d[-1] - d[0])

        # 是否已经经历过接触段
        seen_contact_before = any(
            ph.get("has_contact", False)
            for ph in phases[:phases.index(phase)]
        )

        # hint_phase_type: 由 _segment_five_phases 状态机直接给出，优先级最高
        hint_phase_type = phase.get("hint_phase_type", "")

        # ── 缓存命中 ──────────────────────────────────────────────────────
        if phase_key in cache:
            cached = cache[phase_key]
            # 若状态机已给出 hint，直接用；否则沿用缓存值并做硬校验
            if hint_phase_type:
                phase["phase_type"] = hint_phase_type
            else:
                phase["phase_type"] = _validate_phase_type(
                    cached.get("phase_type", "approach"),
                    has_contact      = has_contact,
                    is_first_contact = is_first_contact,
                    dist_delta       = dist_delta,
                    seen_contact     = seen_contact_before,
                )
            phase["p1_text"] = _fix_self_reference(cached.get("p1_action", ""), "P1")
            phase["p2_text"] = _fix_self_reference(cached.get("p2_action", ""), "P2")
            continue

        contact_hint = (
            "Physical contact occurs in this phase."
            if has_contact else
            "No physical contact in this phase."
        )
        structured = _generate_structured_description(
            sequence, t_start, t_end,
            phase_type=hint_phase_type, has_contact=has_contact,
            executor=executor,
        )
        phase_prompt = (
            f"PHASE: {phase_id}  type={hint_phase_type or '?'}  "
            f"frames [{t_start}, {t_end}]\n"
        )
        if structured and not no_motion_facts:
            phase_prompt += f"MOTION FACTS:\n{structured}\n"
        phase_prompt += "Rewrite as JSON {p1_text, p2_text}:"

        # 追加到多轮 session，复用序列级 KV-cache
        session_messages.append({"role": "user", "content": phase_prompt})

        data = None
        _MAX_ATTEMPTS = 5
        for attempt in range(_MAX_ATTEMPTS):
            raw = _send_messages(session_messages)
            if raw:
                data = _parse_json(raw)
            if data is not None and ("p1_text" in data or "p2_text" in data):
                # 把 assistant 回复追加到 session，供后续 phase 参考
                session_messages.append({"role": "assistant", "content": raw})
                break
            # 重试：移除刚才的失败 user 消息，重新追加；指数退避
            if session_messages and session_messages[-1]["role"] == "user":
                session_messages.pop()
            session_messages.append({"role": "user", "content": phase_prompt})
            wait = 2 ** attempt  # 1, 2, 4, 8 秒
            logger.warning(
                "LLM annotator retry %d/%d for phase %s (wait %ds)",
                attempt + 1, _MAX_ATTEMPTS, phase_id, wait,
            )
            import time as _time
            _time.sleep(wait)

        # 设置 phase_type：优先状态机 hint，其次 LLM，最后硬校验
        if hint_phase_type:
            phase["phase_type"] = hint_phase_type
        elif data is not None:
            phase["phase_type"] = _validate_phase_type(
                data.get("phase_type", "approach"),
                has_contact      = has_contact,
                is_first_contact = is_first_contact,
                dist_delta       = dist_delta,
                seen_contact     = seen_contact_before,
            )
        else:
            # 完全 fallback
            phase["phase_type"] = "contact_hold" if has_contact else (
                "step_back" if seen_contact_before else "approach"
            )

        if data is None:
            # LLM failed: use fallback text for this run but DO NOT write to cache.
            # Leaving the phase uncached ensures the next annotate job will retry it.
            p1_text = "Person 1 makes physical contact with Person 2." if has_contact and executor == "P1" else "Person 1 moves."
            p2_text = "Person 2 makes physical contact with Person 1." if has_contact and executor == "P2" else "Person 2 moves."
            phase["p1_text"] = p1_text
            phase["p2_text"] = p2_text
            # Skip writing to cache — this entry will be retried next time.
            continue
        else:
            phase["p1_text"] = _fix_self_reference(data.get("p1_text", ""), "P1")
            phase["p2_text"] = _fix_self_reference(data.get("p2_text", ""), "P2")

        # ── 写缓存（仅 LLM 实际返回结果时写入）──────────────────────────
        cache[phase_key] = {
            "phase_type" : phase["phase_type"],
            "frame_range": [t_start, t_end],
            "p1_action"  : phase["p1_text"],
            "p2_action"  : phase["p2_text"],
        }

    # ── 持久化缓存 ──────────────────────────────────────────────────────────
    if cache_file:
        try:
            out = {
                "seq_id"      : seq_id,
                "global_text" : global_text,
                "fps"         : sequence.get("fps", 30),
                "local_phases": cache,
            }
            with open(cache_file, "w") as cf:
                _json.dump(out, cf, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to write LLM text cache %s: %s", cache_file, e)

    return phases


def _load_interx_annotations(
    annotation_path: Optional[str],
    seq_id         : str,
) -> List[Dict]:
    """
    读取 Inter-X 官方标注。
    格式：JSON 列表，每项含 frame_start / frame_end / action_description。
    若文件不存在或无法解析，返回空列表。
    """
    if annotation_path is None:
        return []
    try:
        fname = os.path.join(annotation_path, f"{seq_id}.json")
        if not os.path.exists(fname):
            # 尝试 txt 格式
            fname = os.path.join(annotation_path, f"{seq_id}.txt")
            if not os.path.exists(fname):
                return []
            # 简单 txt 格式：每行 "start end description"
            anns = []
            with open(fname) as f:
                for line in f:
                    parts = line.strip().split(None, 2)
                    if len(parts) >= 2:
                        anns.append({
                            "frame_start": int(parts[0]),
                            "frame_end"  : int(parts[1]),
                            "description": parts[2] if len(parts) > 2 else "",
                        })
            return anns
        with open(fname) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("读取标注失败 %s: %s", seq_id, e)
        return []


def _build_candidates_from_annotations(
    annotations : List[Dict],
    mode_labels : np.ndarray,
    sequence    : Dict[str, Any],
) -> List[Dict[str, Any]]:
    """从官方标注段边界构建初始候选 phase。"""
    T        = len(mode_labels)
    seq_id   = sequence["seq_id"]
    candidates = []
    for ann in annotations:
        t_start = int(ann.get("frame_start", ann.get("start", 0)))
        t_end   = int(ann.get("frame_end",   ann.get("end",   t_start + 1)))
        t_end   = min(t_end, T)
        if t_end - t_start <= 0:
            continue
        candidates.append({
            "seq_id"           : seq_id,
            "t_start"          : t_start,
            "t_end"            : t_end,
            "raw_duration"     : t_end - t_start,
            "global_text"      : sequence.get("global_text", ""),
            "has_contact"      : False,    # 由后续 contact 检测更新
            "is_splittable"   : False,
            "from_annotation"  : True,
        })
    return candidates


def _detect_contact_flags(
    p1_joints : np.ndarray,   # (T, J, 3)
    p2_joints : np.ndarray,   # (T, J, 3)
    threshold : float = 0.35,   # 宽松阈值，用于 phase 切分（非训练 label）
    min_frames: int   = MIN_CONTACT_FRAMES,
) -> np.ndarray:
    """
    逐帧检测双人接触，返回 debounced bool 数组 (T,)。

    用 wrist/hand 对 upper-body 的最小距离判断接触：
      P1 wrists  ↔  P2 upper body (neck/head/shoulders/elbows)
      P2 wrists  ↔  P1 upper body
      hand-to-hand

    debounce：连续 min_frames 帧接触才确认为接触段。
    """
    # SMPL-X Jtr joint indices (同 8_window_actions_from_fk_llm.py)
    PELVIS, NECK, HEAD = 0, 12, 15
    L_SHOULDER, R_SHOULDER = 16, 17
    L_ELBOW, R_ELBOW = 18, 19
    L_WRIST, R_WRIST = 20, 21
    UPPER = [NECK, HEAD, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW]
    HANDS = [L_WRIST, R_WRIST]

    T = p1_joints.shape[0]
    raw = np.zeros(T, dtype=bool)

    for t in range(T):
        p1 = p1_joints[t]   # (J, 3)
        p2 = p2_joints[t]   # (J, 3)
        
        # 跳过 NaN 帧
        if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
            continue

        # 双向 wrist-to-upper 最小距离
        p1h = p1[HANDS]   # (2, 3)
        p2h = p2[HANDS]
        p1u = p1[UPPER]   # (6, 3)
        p2u = p2[UPPER]

        def _min_dist(a, b):
            # 防守性检查：ensure a 和 b 不为空
            if a.size == 0 or b.size == 0:
                return float('inf')
            try:
                return float(np.min(np.linalg.norm(a[:, None] - b[None], axis=2)))
            except Exception:
                return float('inf')

        try:
            if (_min_dist(p1h, p2u) < threshold or
                    _min_dist(p2h, p1u) < threshold or
                    _min_dist(p1h, p2h) < threshold):
                raw[t] = True
        except Exception:
            # Skip this frame on error
            pass

    # debounce: 连续 min_frames 帧才算真正接触
    contact = np.zeros(T, dtype=bool)
    run = 0
    for t in range(T):
        if raw[t]:
            run += 1
            if run >= min_frames:
                contact[max(0, t - run + 1):t + 1] = True
        else:
            run = 0

    return contact


def _fill_contact_gaps(
    contact_flags : np.ndarray,   # (T,) bool
    max_gap       : int = 15,     # 短于此帧数的非接触间隙填充为接触
) -> np.ndarray:
    """
    填充接触段之间的短暂间隙，防止舞蹈/长时拥抱被切成多个碎片。
    例如：[1,1,1,0,0,1,1,1] 且 gap=2 ≤ max_gap → [1,1,1,1,1,1,1,1]
    """
    flags = contact_flags.copy()
    T = len(flags)
    i = 0
    while i < T:
        if not flags[i]:
            # 找间隙结束位置
            j = i
            while j < T and not flags[j]:
                j += 1
            gap = j - i
            # 前后都有接触段才填充
            if gap <= max_gap and i > 0 and j < T:
                flags[i:j] = True
            i = j
        else:
            i += 1
    return flags


def _wrist_to_upper_dist(
    p1j : np.ndarray,   # (T, J, 3)
    p2j : np.ndarray,   # (T, J, 3)
) -> np.ndarray:
    """
    逐帧计算双向腕部到对方上身关节的最小距离，返回 (T,) float32。
    取 min(P1_wrists→P2_upper, P2_wrists→P1_upper) 以覆盖双向接触动作。
    """
    UPPER = [12, 15, 16, 17, 18, 19]   # NECK, HEAD, L/R_SHOULDER, L/R_ELBOW
    HANDS = [20, 21]                    # L_WRIST, R_WRIST

    p1h = p1j[:, HANDS, :]             # (T, 2, 3)
    p2h = p2j[:, HANDS, :]
    p1u = p1j[:, UPPER, :]             # (T, 6, 3)
    p2u = p2j[:, UPPER, :]

    # (T, 2, 6, 3) differences → (T, 2, 6) distances
    d_p1h_p2u = np.linalg.norm(
        p1h[:, :, None, :] - p2u[:, None, :, :], axis=-1
    ).min(axis=(1, 2))                 # (T,)
    d_p2h_p1u = np.linalg.norm(
        p2h[:, :, None, :] - p1u[:, None, :, :], axis=-1
    ).min(axis=(1, 2))                 # (T,)

    return np.minimum(d_p1h_p2u, d_p2h_p1u).astype(np.float32)


def _smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Box-filter smoothing for 1-D signal."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def _segment_five_phases(
    mode_labels      : np.ndarray,
    sequence         : Dict[str, Any],
    min_phase_frames : int = MIN_PHASE_FRAMES,
) -> List[Dict[str, Any]]:
    """
    五状态前向状态机，直接切分为细粒度 phase：

      approach → reach → contact_hold → release → step_back

    状态转换由运动信号驱动：
      approach → reach        : wrist_to_upper < REACH_DIST_THRESH
      reach    → contact_hold : contact_flags[t] == True
      contact_hold → release  : contact_flags[t] == False
      release  → step_back    : wrist_to_upper > RELEASE_DIST_THRESH

    每个 candidate 带 hint_phase_type 字段，供 LLM annotator 直接使用。
    短于 min_phase_frames 的段合并到相邻 phase。

    若无 FK 结果，退化到 _build_candidates_from_contact。
    """
    seq_id = sequence["seq_id"]
    T      = len(mode_labels)

    interaction_frames = [t for t in range(T)
                          if mode_labels[t] == InteractionMode.INTERACTION]

    if not interaction_frames:
        # Wave / Chat / Point finger at 等远距离动作不会触发 INTERACTION_MODE，
        # 但整段序列本身就是交互过程，以全序列作为交互窗口。
        logger.info("seq %s: no INTERACTION_MODE frames, using full sequence as interaction window", seq_id)
        t_ia_start = 0
        t_ia_end   = T
    else:
        t_ia_start = interaction_frames[0]
        t_ia_end   = interaction_frames[-1] + 1

    if "p1_joints" not in sequence:
        logger.warning("seq %s: 无 FK 结果，退化为 _build_candidates_from_contact", seq_id)
        return _build_candidates_from_contact(mode_labels, sequence, min_phase_frames)

    p1j = sequence["p1_joints"]   # (T_full, J, 3)
    p2j = sequence["p2_joints"]

    contact_flags = _detect_contact_flags(p1j, p2j)              # (T,) bool
    # 空洞填充：接触段之间的短暂间隙（< min_phase_frames 帧）视为持续接触，
    # 避免舞蹈/长时拥抱被切成多个碎片 contact phase。
    contact_flags = _fill_contact_gaps(contact_flags, max_gap=30)
    w2u_raw       = _wrist_to_upper_dist(p1j, p2j)               # (T,) float32
    w2u           = _smooth(w2u_raw, window=7)                    # smoothed

    # ── 动态检测 approach 起点 ────────────────────────────────────────────────
    # 在 t_ia_start 之前找根距离序列的最后一个局部极大值帧，即双人开始持续
    # 靠近的起点。这样可以跳过序列开头双人静止站立的中性段。
    p1r = sequence["p1_root"]   # (T, 3)
    p2r = sequence["p2_root"]
    root_dist = np.linalg.norm(p1r - p2r, axis=-1)   # (T,)

    # 在 [0, t_ia_start] 内找最后一个局部极大值（根距离开始单调下降的帧）
    # 用平滑后的距离避免噪声干扰
    smooth_dist = _smooth(root_dist, window=9)
    t_approach_start = 0
    for t in range(t_ia_start - 1, 0, -1):
        if smooth_dist[t] > smooth_dist[t - 1] and smooth_dist[t] > smooth_dist[t + 1]:
            t_approach_start = t
            break

    t_seq_start = t_approach_start

    # ── 前向状态机（4 状态）─────────────────────────────────────────────────
    #   approach → contact → release  （接触路径）
    #   approach → non_contact_interact → release  （无接触路径，后处理注入）
    state     = "approach"
    seg_start = t_seq_start
    raw_segs  : List[Tuple[int, int, str]] = []   # (t_start, t_end, phase_type)

    for t in range(t_seq_start, t_ia_end):
        new_state = state
        c = bool(contact_flags[t])

        if state == "approach":
            if c:
                new_state = "contact"
        elif state == "contact":
            if not c:
                new_state = "release"
        # release: terminal — no further transitions

        if new_state != state:
            raw_segs.append((seg_start, t, state))
            seg_start = t
            state     = new_state

    raw_segs.append((seg_start, t_ia_end, state))

    # ── 后处理：无接触序列注入 non_contact_interact ─────────────────────────
    # 若整段 raw_segs 中没有 contact，说明是非接触互动（Wave/Chat/RPS 等）。
    # 用平滑根距离找到：
    #   1. approach 结束点 t_nc_start：根距离不再单调下降的最早帧
    #   2. release 开始点 t_nc_end：根距离开始单调上升的最晚帧
    # 之间的段重标为 non_contact_interact。
    has_contact_in_segs = any(pt == "contact" for _, _, pt in raw_segs)
    if not has_contact_in_segs and len(raw_segs) >= 1:
        root_dist_smooth = _smooth(
            np.linalg.norm(sequence["p1_root"] - sequence["p2_root"], axis=-1),
            window=15,
        )
        seg_t0 = raw_segs[0][0]
        seg_t1 = raw_segs[-1][1]

        t_nc_start = seg_t0
        for t in range(seg_t0, min(seg_t1 - 1, seg_t0 + (seg_t1 - seg_t0) // 2)):
            if root_dist_smooth[t] <= root_dist_smooth[t + 1]:
                t_nc_start = t
                break

        t_nc_end = seg_t1
        for t in range(seg_t1 - 1, max(t_nc_start, seg_t1 - (seg_t1 - seg_t0) // 2), -1):
            if root_dist_smooth[t] <= root_dist_smooth[t - 1]:
                t_nc_end = t
                break

        new_segs: List[Tuple[int, int, str]] = []
        if t_nc_start > seg_t0 + min_phase_frames:
            new_segs.append((seg_t0, t_nc_start, "approach"))
        if t_nc_end > t_nc_start + min_phase_frames:
            new_segs.append((t_nc_start, t_nc_end, "non_contact_interact"))
        if seg_t1 > t_nc_end + min_phase_frames:
            new_segs.append((t_nc_end, seg_t1, "release"))

        if new_segs:
            raw_segs = new_segs
        else:
            raw_segs = [(seg_t0, seg_t1, "non_contact_interact")]

    # ── 合并过短的段到相邻（后）phase ─────────────────────────────────────────
    merged: List[Tuple[int, int, str]] = []
    for ts, te, pt in raw_segs:
        if te - ts < min_phase_frames and merged:
            # extend previous segment to cover this one
            prev_ts, prev_te, prev_pt = merged[-1]
            merged[-1] = (prev_ts, te, prev_pt)
        else:
            merged.append((ts, te, pt))

    # ── 构建 candidate dicts ──────────────────────────────────────────────────
    global_text = sequence.get("global_text", "")
    candidates  = []
    for ts, te, pt in merged:
        if te - ts < min_phase_frames:
            continue
        candidates.append({
            "seq_id"          : seq_id,
            "t_start"         : ts,
            "t_end"           : te,
            "raw_duration"    : te - ts,

            "global_text"     : global_text,
            "has_contact"     : pt == "contact",
            "hint_phase_type" : pt,
            "is_splittable"   : False,
            "from_annotation" : False,
        })

    if not candidates:
        # 完全退化
        candidates.append({
            "seq_id"          : seq_id,
            "t_start"         : t_seq_start,
            "t_end"           : t_ia_end,
            "raw_duration"    : t_ia_end - t_seq_start,

            "global_text"     : global_text,
            "has_contact"     : False,
            "hint_phase_type" : "non_contact_interact",
            "is_splittable"   : False,
            "from_annotation" : False,
        })

    return candidates


def _build_candidates_from_contact(
    mode_labels : np.ndarray,
    sequence    : Dict[str, Any],
    min_phase_frames: int = MIN_PHASE_FRAMES,
) -> List[Dict[str, Any]]:
    """
    方法2：接触事件驱动的语义 phase 切分。

    在 INTERACTION_MODE 范围内，按接触段边界切分：
      - walk_towards 段（进入 INTERACTION 到首次接触前）→ 一个 phase
      - 每段接触（contact_flags=True 的连续段）→ 一个 phase
      - 接触段之间的间隔（短暂分离）→ 若 < min_phase_frames 则合入前段

    需要 sequence 中包含 p1_joints / p2_joints（FK 结果）。
    若无 FK 结果，退化为整段 INTERACTION 作为一个 phase。
    """
    T      = len(mode_labels)
    seq_id = sequence["seq_id"]

    # 找 INTERACTION_MODE 范围
    interaction_frames = [t for t in range(T)
                          if mode_labels[t] == InteractionMode.INTERACTION]
    if not interaction_frames:
        return []

    t_ia_start = interaction_frames[0]
    t_ia_end   = interaction_frames[-1] + 1

    # 若无 FK 结果，退化为整段
    if "p1_joints" not in sequence:
        logger.warning("seq %s: 无 FK 结果，退化为整段 INTERACTION phase", seq_id)
        return [{
            "seq_id"         : seq_id,
            "t_start"        : t_ia_start,
            "t_end"          : t_ia_end,
            "raw_duration"   : t_ia_end - t_ia_start,

            "global_text"    : sequence.get("global_text", ""),
            "has_contact"    : False,
            "is_splittable"   : False,
            "from_annotation": False,
        }]

    # 接触检测（仅在 INTERACTION 段内）
    p1j = sequence["p1_joints"]   # (T_full, J, 3)
    p2j = sequence["p2_joints"]
    contact_flags = _detect_contact_flags(p1j, p2j)   # (T_full,)

    # 在 INTERACTION 范围内切分
    candidates = []
    global_text = sequence.get("global_text", "")

    # 找接触段边界
    in_contact = False
    seg_start  = t_ia_start

    def _add(t_s, t_e, has_c):
        if t_e - t_s >= min_phase_frames:
            candidates.append({
                "seq_id"         : seq_id,
                "t_start"        : t_s,
                "t_end"          : t_e,
                "raw_duration"   : t_e - t_s,
    
                "global_text"    : global_text,
                "has_contact"    : has_c,
                "is_splittable"  : False,
                "from_annotation": False,
            })

    for t in range(t_ia_start, t_ia_end):
        c = bool(contact_flags[t])
        if c and not in_contact:
            # 接触开始：先把 walk_towards / 间隔段关掉
            if t > seg_start:
                _add(seg_start, t, has_c=False)
            seg_start  = t
            in_contact = True
        elif not c and in_contact:
            # 接触结束
            _add(seg_start, t, has_c=True)
            seg_start  = t
            in_contact = False

    # 收尾
    if seg_start < t_ia_end:
        _add(seg_start, t_ia_end, has_c=in_contact)

    # 若切分结果为空（整段无接触），退化为整段
    if not candidates:
        _add(t_ia_start, t_ia_end, has_c=False)

    return candidates


def _build_candidates_from_distance(
    mode_labels : np.ndarray,
    sequence    : Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    主入口：调用五状态状态机做细粒度 phase 切分。
    无 FK 时退化到 _build_candidates_from_contact。
    """
    return _segment_five_phases(mode_labels, sequence)


def _merge_adjacent_phases(
    candidates : List[Dict[str, Any]],
    sequence   : Dict[str, Any],
    gap_thresh : int   = 5,
    max_merged : int   = 120,
) -> List[Dict[str, Any]]:
    """
    合并相邻 phase：
      - gap < gap_thresh 帧
      - 整段过渡期内双人距离 < EXIT_DIST
      - 合并后总帧数 ≤ max_merged
    """
    if not candidates:
        return []

    p1_roots = sequence["p1_root"]
    p2_roots = sequence["p2_root"]

    merged = [candidates[0].copy()]
    for curr in candidates[1:]:
        prev = merged[-1]
        gap  = curr["t_start"] - prev["t_end"]
        total = curr["t_end"] - prev["t_start"]

        # 不合并 has_contact 不同的相邻段（保留语义切分边界）
        if prev.get("has_contact") != curr.get("has_contact"):
            merged.append(curr.copy())
            continue

        if gap < gap_thresh and total <= max_merged:
            # 检查过渡期距离
            ok = True
            for t in range(prev["t_end"], curr["t_start"]):
                t = min(t, len(p1_roots) - 1)
                d = np.linalg.norm(p1_roots[t, [0, 2]] - p2_roots[t, [0, 2]])
                if d > EXIT_DIST:
                    ok = False
                    break
            if ok:
                merged[-1]["t_end"]        = curr["t_end"]
                merged[-1]["raw_duration"] = merged[-1]["t_end"] - merged[-1]["t_start"]
                merged[-1]["is_splittable"] = False
                continue
        merged.append(curr.copy())

    return merged


def _all_in_interaction(
    phase_meta  : Dict[str, Any],
    mode_labels : np.ndarray,
) -> bool:
    t_start = phase_meta["t_start"]
    t_end   = min(phase_meta["t_end"], len(mode_labels))
    if t_start >= t_end:
        return False
    return all(
        mode_labels[t] == InteractionMode.INTERACTION
        for t in range(t_start, t_end)
    )


# ── contact_frame_mask extraction ────────────────────────────────────────────

def extract_contact_frame_mask(
    p1_joints        : np.ndarray,            # [T, J, 3]
    p2_joints        : np.ndarray,            # [T, J, 3]
    contact_pairs_idx: List[Tuple[int, int]], # [(p1_j_idx, p2_j_idx)]
    t_start          : int,
    t_end            : int,
    dist_thresh      : float = CONTACT_DIST_THRESH,
    min_frames       : int   = MIN_CONTACT_FRAMES,
    windup_frames    : int   = REACH_WINDUP_FRAMES,
) -> np.ndarray:
    """
    从 GT 骨架关节对距离自动推导 contact_frame_mask [W]。

    算法：
      1. 关节对距离计算
      2. 初始阈值过滤
      3. 形态学开运算去抖（连续 True ≥ min_frames）
      4. 前摇清零（每段 True 起始前 windup_frames 帧置 False）
    """
    W = t_end - t_start
    if not contact_pairs_idx:
        return np.zeros(W, dtype=bool)

    # 1. 逐帧计算最小距离
    raw_mask = np.zeros(W, dtype=bool)
    for k, (i, j) in enumerate(contact_pairs_idx):
        for t_local in range(W):
            t_global = t_start + t_local
            d = np.linalg.norm(
                p1_joints[t_global, i] - p2_joints[t_global, j]
            )
            if d < dist_thresh:
                raw_mask[t_local] = True

    # 2. 形态学开运算：连续 True 段长度 < min_frames 时删除
    opened = np.zeros(W, dtype=bool)
    i = 0
    while i < W:
        if raw_mask[i]:
            j = i
            while j < W and raw_mask[j]:
                j += 1
            if j - i >= min_frames:
                opened[i:j] = True
            i = j
        else:
            i += 1

    # 3. 前摇清零
    result = opened.copy()
    i = 0
    while i < W:
        if opened[i] and (i == 0 or not opened[i - 1]):
            # 找到段起始
            clear_start = max(0, i - windup_frames)
            result[clear_start:i] = False
        i += 1

    return result


# ── Ego-centric transform for a phase ────────────────────────────────────────

def _joints_to_egocentric(
    joints_world : np.ndarray,  # [T, J, 3]
    t_start      : int,
    ref_root_pos : np.ndarray,  # [3]  t_start 处参考 root
    ref_root_rot : np.ndarray,  # [4]  t_start 处参考 root 旋转
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将整段关节序列变换到以 t_start 为原点的 Ego-centric 坐标。

    同时应用 Y-grounding（使 t_start 处脚部最低点对齐到 Y=0）。
    返回：
      joints_local : [T, J, 3]
      origin       : [3]  参考平移
      R_inv        : [3, 3] 参考旋转逆矩阵
    """
    origin, R_inv = build_egocentric_transform(ref_root_pos, ref_root_rot)
    joints_local  = apply_egocentric_transform(joints_world, origin, R_inv)
    return joints_local, origin, R_inv


# ── HunyuanMotion 201-dim motion representation ───────────────────────────────
#
# Layout (per frame):
#   [0:3]    root XYZ position   (ego-centric)
#   [3:9]    root 6D rotation    (ego-centric yaw-canonical; joint 0 world-rot · R_inv)
#   [9:135]  joints 1-21 6D rotation  (local parent-relative, no ego-centric)
#   [135:201] joints 0-21 3D position (ego-centric, 22×3 = 66)
#
# 6D rotation = first two columns of a 3×3 rotation matrix, concatenated.

def _aa_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) → rotation matrices (..., 3, 3) via scipy."""
    from scipy.spatial.transform import Rotation
    orig_shape = aa.shape[:-1]
    mats = Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix()
    return mats.reshape(*orig_shape, 3, 3).astype(np.float32)


def _rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (..., 3, 3) → 6D in HunyuanMotion format.

    HunyuanMotion's rot6d_to_rotation_matrix interprets 6 values as a (3, 2)
    row-major matrix: [R00, R01, R10, R11, R20, R21].
    Column-concat format [col0; col1] would give [R00,R10,R20,R01,R11,R21],
    which is a different permutation and causes completely wrong reconstructed rotations.
    """
    c0 = R[..., :, 0]   # (..., 3) = [R00, R10, R20]
    c1 = R[..., :, 1]   # (..., 3) = [R01, R11, R21]
    return np.stack(
        [c0[..., 0], c1[..., 0],   # R00, R01
         c0[..., 1], c1[..., 1],   # R10, R11
         c0[..., 2], c1[..., 2]],  # R20, R21
        axis=-1,
    ).astype(np.float32)


def _build_hy201_seq(
    smplx_params   : np.ndarray,   # (T_full, 56, 3) full-seq axis-angle
    joints_grounded: np.ndarray,   # (T_full, 55, 3) full-seq world-space grounded joints
    t_start        : int,          # slice start (may be negative → zero-pad left)
    t_end          : int,          # slice end
    target_W       : int,          # output length
    origin         : np.ndarray,   # (3,) ego-centric translation offset
    R_inv          : np.ndarray,   # (3, 3) ego-centric inverse yaw rotation
) -> np.ndarray:
    """Build (target_W, 201) motion feature tensor in HunyuanMotion format.

    Handles zero-padding on the left (when t_start < 0) and repeat-last-frame
    padding on the right (when the slice is shorter than target_W).
    """
    T_full   = smplx_params.shape[0]
    t_s      = max(0, t_start)
    t_e      = min(t_end, T_full)
    L        = max(0, t_e - t_s)
    pad_left = max(0, -t_start)   # frames before seq start → zeros

    if L > 0:
        aa_slice  = smplx_params[t_s:t_e]             # (L, 56, 3)
        pos_slice = joints_grounded[t_s:t_e, :22, :]  # (L, 22, 3) joints 0-21

        # Root rotation (joint 0): world → ego-centric yaw → 6D
        root_R_world = _aa_to_rotmat(aa_slice[:, 0, :])              # (L, 3, 3)
        root_R_ego   = np.einsum('ij,tjk->tik', R_inv, root_R_world) # (L, 3, 3)
        root_6d      = _rotmat_to_6d(root_R_ego)                      # (L, 6)

        # Root position: ego-centric
        root_pos_ego = np.einsum(
            'ij,tj->ti', R_inv, pos_slice[:, 0, :] - origin
        )  # (L, 3)

        # Body joints 1-21: local axis-angle → rotation matrix → 6D
        body_R  = _aa_to_rotmat(
            aa_slice[:, 1:22, :].reshape(-1, 3)
        ).reshape(L, 21, 3, 3)
        body_6d = _rotmat_to_6d(body_R)   # (L, 21, 6)

        # All joint positions 0-21: root-relative in ego-centric frame.
        # Subtract current-frame root so FK[0] = [0,0,0] always.
        # Separates pose (FK) from global translation ([0:3]), matching HY convention.
        # NOTE: when using partner motion as V3 conditioning, explicitly reconstruct
        # absolute positions: partner_fk_abs = partner_root + partner_fk_rel
        joint_pos_ego_abs = np.einsum(
            'ij,tlj->tli', R_inv, pos_slice - origin
        )  # (L, 22, 3)
        joint_pos_ego = joint_pos_ego_abs - joint_pos_ego_abs[:, 0:1, :]  # (L, 22, 3) root-relative

        feat = np.concatenate([
            root_pos_ego,                        # (L, 3)
            root_6d,                             # (L, 6)
            body_6d.reshape(L, 21 * 6),          # (L, 126)
            joint_pos_ego.reshape(L, 22 * 3),    # (L, 66)
        ], axis=-1).astype(np.float32)           # (L, 201)
    else:
        feat = np.zeros((0, 201), dtype=np.float32)

    # Left-pad with zeros for frames before sequence start
    if pad_left > 0:
        feat = np.concatenate(
            [np.zeros((pad_left, 201), dtype=np.float32), feat], axis=0
        )

    total = pad_left + L
    if total < target_W:
        shortage = target_W - total
        if total > 0:
            feat = np.concatenate(
                [feat, np.repeat(feat[-1:], shortage, axis=0)], axis=0
            )
        else:
            feat = np.zeros((target_W, 201), dtype=np.float32)
    else:
        feat = feat[:target_W]

    return feat   # (target_W, 201)


# ── build_phase_sample ────────────────────────────────────────────────────────

def build_phase_sample(
    actor_id          : str,           # "P1" | "P2"
    phase_meta        : Dict[str, Any],
    sequence          : Dict[str, Any],
    foot_joint_indices: List[int],
    phase_text             : str = "",         # natural language description of this actor's action
    phase_type             : str = "approach", # closed-set semantic label
    interaction_category   : str = "",         # Inter-X action name (e.g. "Hug")
    prev_phase_meta   : Optional[Dict[str, Any]] = None,  # 上一个 phase 的 meta（P1 rollout 用）
    phase0_meta       : Optional[Dict[str, Any]] = None,  # 第一个 phase 的 meta（initial_frame 用）
                                                           # None 表示当前 phase 本身就是 Phase 0
) -> PhaseSample:
    """
    构建单个 actor 在单个 phase 的训练样本（PhaseSample）。

    严格遵循 Ego-centric 约定：
      - 参考帧为 phase t_start 的执行 actor root
      - self 和 partner 使用完全同一组刚体变换

    partner_rollout_local 约定（run_017 数据集）：
      - Phase 0 P1/P2（对称）：partner_f0 × W（对方 phase0 首帧重复 W 次）
      - Phase 1+ P1         ：P2 上一 phase GT，长度 prev_W（时序 Ping-Pong）
      - Phase 1+ P2         ：P1 当前 phase GT，长度 W（teacher forcing）
        推理时由 inference loop 替换为 P1 实际预测输出

    phase_text: natural language description of this actor's role in this phase.
      Stored in PhaseSample for later encoding by the text encoder.
    """
    seq_id   = phase_meta["seq_id"]
    phase_id = phase_meta["phase_id"]
    t_start  = phase_meta["t_start"]
    t_end    = phase_meta["t_end"]
    W        = phase_meta["duration_bucket"]  # 含 tail fill 的 bucket 长度

    # ── 确定 self/partner 数据通道 ───────────────────────────────────────────
    if actor_id == "P1":
        self_joints_world    = sequence["p1_joints"]      # [T_full, J, 3]
        self_smplx_params    = sequence["p1_smplx_params"]  # [T_full, 56, 3]
        partner_joints_world = sequence["p2_joints"]
        partner_smplx_params = sequence["p2_smplx_params"]
    else:
        self_joints_world    = sequence["p2_joints"]
        self_smplx_params    = sequence["p2_smplx_params"]
        partner_joints_world = sequence["p1_joints"]
        partner_smplx_params = sequence["p1_smplx_params"]

    T_full = self_joints_world.shape[0]

    # ── Y-grounding（预处理，训练数据与推理保持一致）──────────────────────────
    self_joints_grounded    = apply_y_grounding_sequence(self_joints_world, foot_joint_indices)
    partner_joints_grounded = apply_y_grounding_sequence(partner_joints_world, foot_joint_indices)

    # ── 构建 Ego-centric 变换（以 t_start 的 self root 为参考）───────────────
    # Use actual root yaw from smplx_params (joint 0 axis-angle) for proper alignment.
    t_ref = min(t_start, T_full - 1)
    _root_aa_ref = self_smplx_params[t_ref, 0, :]
    _root_R_ref  = _aa_to_rotmat(_root_aa_ref[None])[0]   # (3, 3)
    _yaw_ref     = _extract_root_yaw(_root_R_ref)
    _R_ref       = _yaw_rotation_matrix(_yaw_ref)
    R_inv        = _R_ref.T                                # (3, 3)
    R            = _R_ref                                  # forward rotation

    # Origin: grounded pelvis position at t_start (joint 0 after grounding).
    # Only remove horizontal position (X, Z); keep absolute Y so that root
    # translation [0:3] retains ground-relative height (≈0.9m for standing),
    # matching HunyuanMotion's pretrained convention.
    # R_inv is a pure Y-axis rotation → Y component is unaffected by R_inv,
    # so zeroing origin.Y makes ego_Y = world_Y (absolute height). ✓
    origin = self_joints_grounded[t_ref, 0, :].copy()     # (3,)
    origin[1] = 0.0   # keep absolute Y in ego frame

    # ── real_len & W ─────────────────────────────────────────────────────────
    real_len = min(t_end - t_start, T_full - t_start)
    W        = phase_meta["duration_bucket"]

    # Safety check: if real_len > W, bucket is wrong → snap up
    if real_len > W:
        logger.warning(
            "phase %s actor=%s: real_len=%d > bucket=%d, recalculating bucket",
            phase_id, actor_id, real_len, W
        )
        W = _nearest_bucket(real_len)
        phase_meta["duration_bucket"] = W

    # ── self_history_local [W_PREV, 201] ─────────────────────────────────────
    self_hist_local = _build_hy201_seq(
        self_smplx_params, self_joints_grounded,
        t_start - W_PREV, t_start, W_PREV, origin, R_inv,
    )  # (W_PREV, 201)

    # ── partner_history_local [W_PREV, 201] ──────────────────────────────────
    partner_hist_local = _build_hy201_seq(
        partner_smplx_params, partner_joints_grounded,
        t_start - W_PREV, t_start, W_PREV, origin, R_inv,
    )  # (W_PREV, 201)

    # ── target_motion_local [W, 201]（含 tail fill）──────────────────────────
    target_local = _build_hy201_seq(
        self_smplx_params, self_joints_grounded,
        t_start, t_start + real_len, W, origin, R_inv,
    )  # (W, 201)

    # ── seq_mask [W] ─────────────────────────────────────────────────────────
    seq_mask = np.zeros(W, dtype=bool)
    seq_mask[:real_len] = True

    # ── initial_frame & partner_initial_frame ────────────────────────────────
    # initial_frame: self's phase0 first frame in CURRENT phase's ego frame
    # partner_initial_frame: partner's phase0 first frame in CURRENT phase's ego frame
    # (same rigid transform as current phase; root position ≠ 0 for phase1+)
    ph0_ts = min(
        phase0_meta["t_start"] if phase0_meta is not None else t_start,
        T_full - 1,
    )
    initial_frame = _build_hy201_seq(
        self_smplx_params, self_joints_grounded, ph0_ts, ph0_ts + 1, 1, origin, R_inv
    )[0]  # (201,)
    partner_initial_frame_arr = _build_hy201_seq(
        partner_smplx_params, partner_joints_grounded, ph0_ts, ph0_ts + 1, 1, origin, R_inv
    )[0]  # (201,)

    # ── prefix_mask [W] ──────────────────────────────────────────────────────
    # 所有 phase（含 Phase 0）均以前 OVERLAP_N 帧作为条件前缀：
    #   Phase 0   ：content = frame0 × OVERLAP_N（dataset 加载时由 __getitem__ 填入）
    #   Phase 1+ ：content = 上一 phase 末尾 OVERLAP_N 帧（inpainting）
    prefix_mask = np.zeros(W, dtype=bool)
    prefix_mask[:OVERLAP_N] = True

    # ── partner_prev_local / partner_curr_local ───────────────────────────────
    # 两字段分别对应时序 Ping-Pong 的两种用途，均以 self 当前 phase ego 坐标系表示。
    # 训练/推理代码按 actor_id 选择读取哪个字段：
    #   P1 (Ping) → partner_prev_local；P2 (Pong) → partner_curr_local
    history_ids = np.arange(-W_PREV, 0, dtype=np.int32)
    rollout_ids = np.arange(0,       W,  dtype=np.int32)

    def _build_rollout_hy201(smplx_p: np.ndarray, joints_g: np.ndarray,
                              rs: int, re: int, target_W: int) -> np.ndarray:
        return _build_hy201_seq(smplx_p, joints_g, rs, re, target_W, origin, R_inv)

    is_phase0 = (phase0_meta is None)   # True iff current phase is Phase 0

    # ── partner_prev_local ────────────────────────────────────────────────────
    if is_phase0:
        # Phase 0：无上一 phase，用 partner_f0 × W 作为对称 proxy
        partner_prev_local    = np.tile(partner_initial_frame_arr[np.newaxis, :], (W, 1))
        partner_prev_seq_mask = np.ones(W, dtype=bool)
        prev_rollout_ids      = np.zeros(W, dtype=np.int32)
    else:
        prev_ts   = prev_phase_meta["t_start"]
        prev_te   = prev_phase_meta["t_end"]
        prev_W    = prev_phase_meta["duration_bucket"]
        partner_prev_local    = _build_rollout_hy201(
            partner_smplx_params, partner_joints_grounded, prev_ts, prev_te, prev_W
        )
        prev_real_len         = min(prev_te - prev_ts, T_full - prev_ts)
        partner_prev_seq_mask = np.arange(prev_W) < prev_real_len
        prev_rollout_ids      = np.arange(-prev_W, 0, dtype=np.int32)

    partner_prev_time_ids = np.concatenate([history_ids, prev_rollout_ids])

    # ── partner_curr_local ────────────────────────────────────────────────────
    if is_phase0:
        # Phase 0：同样用 partner_f0 × W（对称训练；无 GT rollout 可用）
        partner_curr_local    = np.tile(partner_initial_frame_arr[np.newaxis, :], (W, 1))
        partner_curr_seq_mask = np.ones(W, dtype=bool)
    else:
        partner_curr_local    = _build_rollout_hy201(
            partner_smplx_params, partner_joints_grounded, t_start, t_start + real_len, W
        )
        partner_curr_seq_mask = seq_mask.copy()

    partner_curr_time_ids = np.concatenate([history_ids, rollout_ids])

    # ── contact_frame_mask [W] ────────────────────────────────────────────────
    contact_full = _detect_contact_flags(
        self_joints_grounded if actor_id == "P1" else partner_joints_grounded,
        partner_joints_grounded if actor_id == "P1" else self_joints_grounded,
    )
    cfm = contact_full[t_start : t_start + real_len].astype(bool)
    if len(cfm) < W:
        cfm = np.concatenate([cfm, np.zeros(W - len(cfm), dtype=bool)])

    # ── Assemble PhaseSample ──────────────────────────────────────────────────
    sample = PhaseSample(
        sequence_id       = seq_id,
        phase_id          = phase_id,
        actor_id          = actor_id,
        phase_len_bucket  = W,
        mode              = InteractionMode.INTERACTION,

        self_history_local    = self_hist_local,        # (W_PREV, 201)
        partner_history_local = partner_hist_local,     # (W_PREV, 201)
        partner_prev_local    = partner_prev_local,     # (W_prev, 201)
        partner_prev_seq_mask = partner_prev_seq_mask,  # (W_prev,) bool
        partner_curr_local    = partner_curr_local,     # (W, 201)
        partner_curr_seq_mask = partner_curr_seq_mask,  # (W,) bool
        target_motion_local   = target_local,           # (W, 201)
        partner_prev_time_ids = partner_prev_time_ids,  # (W_PREV+W_prev,) int32
        partner_curr_time_ids = partner_curr_time_ids,  # (W_PREV+W,) int32

        initial_frame         = initial_frame,          # (201,)
        partner_initial_frame = partner_initial_frame_arr,  # (201,)

        phase_type             = phase_type,
        interaction_category   = interaction_category,
        phase_text             = phase_text,
        contact_frame_mask     = cfm,
        seq_mask          = seq_mask,
        prefix_mask       = prefix_mask,

        t_start           = t_start,
        t_end             = t_end,
        partner_source    = PartnerSource.GT,
        is_split_child    = phase_meta.get("is_split_child", False),
        split_parent_id   = phase_meta.get("split_parent_id"),
    )
    return sample


# ── Export dataset ────────────────────────────────────────────────────────────

def export_dataset(
    split_name          : str,
    output_dir          : str,
    h5_path             : str = INTERX_H5_PATH,
    split_root          : str = INTERX_SPLIT_ROOT,
    annotation_dir      : Optional[str] = None,
    cache_path          : Optional[str] = None,
    keyword_map_path    : Optional[str] = None,
    foot_joint_indices  : Optional[List[int]] = None,
    body_model          = None,
    fk_device           : str = "cpu",
    use_llm_annotation  : bool = False,
    llm_model           : str = "qwen3.5:35b",
    llm_base_url        : Optional[str] = None,
    llm_schema_cache_dir: Optional[str] = None,
) -> None:
    """
    完整数据集导出流程：
      1. 按官方 split 加载 sequence id 列表
      2. 逐条序列回放打标 + phase 切分
      3. （可选）LLM schema 标注：use_llm_annotation=True 时调用 LLM，
         为每个 phase 生成 P1/P2 自然语言描述，结果缓存在 llm_schema_cache_dir
      4. 构建 Ping/Pong 样本对
      5. 序列化到 .npz 并导出 dataset_stats.json

    输出：
      {output_dir}/{split_name}/  下若干 .npz 文件（每个 phase 一个）
      {output_dir}/dataset_stats.json

    use_llm_annotation=False（默认）：schema 回退到 annotation_text 关键词解析
      或 [N/A] 占位符，适合快速调试。
    use_llm_annotation=True：LLM 标注，训练/推理 schema 分布一致，推荐用于
      正式预处理。llm_schema_cache_dir 指定缓存目录（避免重复调用 LLM）。
    """
    os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)

    seq_ids = load_split_ids(split_name, split_root)
    logger.info("split=%s: %d sequences", split_name, len(seq_ids))

    stats = {
        "split"            : split_name,
        "n_sequences"      : len(seq_ids),
        "n_phases"         : 0,
        "n_samples"        : 0,
        "bucket_dist"      : {},
        "fallback_seqs"    : 0,
        "llm_annotation"   : use_llm_annotation,
    }

    for seq_id in seq_ids:
        try:
            sequence = load_sequence_from_h5(seq_id, h5_path)
            sequence = compute_world_joints(sequence, body_model=body_model, device=fk_device)
        except Exception as e:
            logger.error("skip seq %s: %s", seq_id, e)
            continue

        mode_labels, sm = replay_and_label_modes(sequence, foot_joint_indices)

        ann_path = os.path.join(annotation_dir, seq_id) if annotation_dir else None
        phases   = segment_interaction_phases(
            sequence, mode_labels,
            annotation_path=ann_path,
            cache_path=cache_path,
            keyword_map_path=keyword_map_path,
        )

        # ── LLM schema annotation（可选）─────────────────────────────────
        if use_llm_annotation and phases:
            try:
                phases = annotate_phases_with_llm(
                    phases,
                    sequence,
                    llm_model=llm_model,
                    llm_base_url=llm_base_url,
                    llm_cache_path=llm_schema_cache_dir,
                )
            except Exception as e:
                logger.error(
                    "LLM annotation failed for seq %s (使用 fallback schema): %s",
                    seq_id, e,
                )

        interaction_category = _action_name_from_seq_id(seq_id)
        prev_phase_meta  = None
        seq_phase0_meta  = None   # 每条序列的第一个 phase meta（用于 initial_frame）
        for phase_meta in phases:
            stats["n_phases"] += 1
            bucket = phase_meta.get("duration_bucket", 0)
            stats["bucket_dist"][str(bucket)] = stats["bucket_dist"].get(str(bucket), 0) + 1

            for actor_id in ["P1", "P2"]:
                # LLM 标注的自然语言描述（use_llm_annotation=True 时由 annotate_phases_with_llm 填入）
                text_key   = "p1_text" if actor_id == "P1" else "p2_text"
                phase_text = phase_meta.get(text_key, "")

                try:
                    sample = build_phase_sample(
                        actor_id=actor_id,
                        phase_meta=phase_meta,
                        prev_phase_meta=prev_phase_meta,
                        phase0_meta=seq_phase0_meta,
                        sequence=sequence,
                        foot_joint_indices=foot_joint_indices or [],
                        phase_text=phase_text,
                        phase_type=phase_meta.get("phase_type", "approach"),
                        interaction_category=interaction_category,
                    )
                    fname = os.path.join(
                        output_dir, split_name,
                        f"{sample.phase_id}_{actor_id}.npz",
                    )
                    _save_sample(sample, fname)
                    stats["n_samples"] += 1
                except Exception as e:
                    logger.error(
                        "build_phase_sample failed seq=%s phase=%s actor=%s: %s",
                        seq_id, phase_meta.get("phase_id"), actor_id, e,
                    )

            if seq_phase0_meta is None:
                seq_phase0_meta = phase_meta  # 记录首 phase（Phase 0 处理完后生效）
            prev_phase_meta = phase_meta  # 供下一个 phase 的 P1 使用

    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("dataset_stats written to %s", stats_path)

    # ── 预处理阶段统计 phase bucket table ─────────────────────────────────────
    # 仅对 train split 统计（val/test 不参与）
    if split_name == "train":
        try:
            from llm_interaction.hhi_planner import compute_interaction_phase_bucket_table
            table_path = os.path.join(output_dir, "phase_bucket_table.json")
            compute_interaction_phase_bucket_table(
                data_dir  = output_dir,
                split     = split_name,
                save_path = table_path,
            )
        except Exception as e:
            logger.warning("phase_bucket_table 统计失败（不影响数据集导出）: %s", e)


def _save_sample(sample: PhaseSample, path: str) -> None:
    """将 PhaseSample 序列化为 .npz（仅保存 ndarray 字段）。"""
    data = {
        "sequence_id"     : np.array([sample.sequence_id]),
        "phase_id"        : np.array([sample.phase_id]),
        "actor_id"        : np.array([sample.actor_id]),
        "phase_len_bucket": np.array([sample.phase_len_bucket]),
        "t_start"         : np.array([sample.t_start]),
        "t_end"           : np.array([sample.t_end]),
        "phase_type"           : np.array([sample.phase_type]),
        "interaction_category" : np.array([sample.interaction_category]),
        "phase_text"           : np.array([sample.phase_text]),
        "global_text"          : np.array([sample.global_text]),
    }
    for field_name in [
        "self_history_local", "partner_history_local",
        "target_motion_local",
        "contact_frame_mask", "seq_mask", "prefix_mask",
        # partner prev-phase conditioning (P1 Ping 使用)
        "partner_prev_local", "partner_prev_seq_mask", "partner_prev_time_ids",
        # partner curr-phase conditioning (P2 Pong 使用)
        "partner_curr_local", "partner_curr_seq_mask", "partner_curr_time_ids",
        # global initial frames
        "initial_frame", "partner_initial_frame",
    ]:
        val = getattr(sample, field_name, None)
        if val is not None:
            data[field_name] = val

    np.savez_compressed(path, **data)


# ── H5 metadata export ────────────────────────────────────────────────────────

def export_metadata_h5(
    output_dir       : str,
    split_name       : str,
    phase_metadata   : List[Dict],
    conflict_logs    : Dict[str, List[dict]],
    sm_logs          : Dict[str, List[dict]],
    duration_logs    : Dict[str, List[dict]],
    split_manifest   : Dict[str, List[str]],
) -> str:
    """
    将数据集构建过程的完整 Metadata 写入 HDF5 压缩文件。

    文件布局：
      /split_manifest/{split_name}   : 每 split 的 sequence id 列表
      /phases/{phase_id}/            : 每 phase 的元信息组
        .attrs: t_start, t_end, duration_bucket, is_split_child, split_parent_id, seq_id
      /conflict_logs/{phase_id}      : 冲突裁决日志 JSON 字节串
      /sm_logs/{seq_id}              : 状态机 entry/exit 距离和接触日志 JSON
      /duration_logs/{phase_id}      : DurationBucketResolver 裁决日志 JSON

    返回 HDF5 文件路径。
    """
    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, f"metadata_{split_name}.h5")

    import json as _json

    with h5py.File(h5_path, "w") as f:
        # ── split manifest ────────────────────────────────────────────────────
        mf_grp = f.require_group("split_manifest")
        for sname, seq_list in split_manifest.items():
            dt = h5py.special_dtype(vlen=str)
            ds = mf_grp.create_dataset(
                sname, shape=(len(seq_list),), dtype=dt, compression="gzip"
            )
            for i, sid in enumerate(seq_list):
                ds[i] = sid

        # ── phase metadata ────────────────────────────────────────────────────
        ph_grp = f.require_group("phases")
        for pm in phase_metadata:
            pid = pm.get("phase_id", "unknown")
            grp = ph_grp.require_group(pid)
            for attr_key in [
                "t_start", "t_end", "duration_bucket",
                "is_split_child", "seq_id",
            ]:
                if attr_key in pm:
                    grp.attrs[attr_key] = pm[attr_key]
            if pm.get("split_parent_id") is not None:
                grp.attrs["split_parent_id"] = pm["split_parent_id"]
            # store full dict as JSON blob for completeness
            grp.attrs["_json"] = _json.dumps(pm, default=str)

        # ── conflict logs ─────────────────────────────────────────────────────
        cl_grp = f.require_group("conflict_logs")
        for phase_id, log_entries in conflict_logs.items():
            cl_grp.create_dataset(
                phase_id,
                data=np.bytes_(_json.dumps(log_entries, ensure_ascii=False)),
                compression="gzip",
            )

        # ── state machine logs ────────────────────────────────────────────────
        sm_grp = f.require_group("sm_logs")
        for seq_id, events in sm_logs.items():
            sm_grp.create_dataset(
                seq_id,
                data=np.bytes_(_json.dumps(events, ensure_ascii=False, default=str)),
                compression="gzip",
            )

        # ── duration resolver logs ────────────────────────────────────────────
        dur_grp = f.require_group("duration_logs")
        for phase_id, records in duration_logs.items():
            dur_grp.create_dataset(
                phase_id,
                data=np.bytes_(_json.dumps(records, ensure_ascii=False, default=str)),
                compression="gzip",
            )

    logger.info("metadata H5 written to %s", h5_path)
    return h5_path


def export_dataset_with_metadata(
    split_name       : str,
    output_dir       : str,
    h5_path          : str = INTERX_H5_PATH,
    split_root       : str = INTERX_SPLIT_ROOT,
    annotation_dir   : Optional[str] = None,
    cache_path       : Optional[str] = None,
    keyword_map_path : Optional[str] = None,
    foot_joint_indices: Optional[List[int]] = None,
    body_model       = None,
    fk_device        : str = "cpu",
) -> None:
    """
    完整数据集导出（含 H5 metadata）：
      - npz 样本文件（与 export_dataset 相同）
      - metadata_{split_name}.h5（conflict logs, sm logs, duration logs, split manifest）
      - dataset_stats.json（含 fallback_segs 比例）
    """
    os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)

    all_splits = ["train", "val", "test"]
    split_manifest: Dict[str, List[str]] = {}
    for s in all_splits:
        try:
            split_manifest[s] = load_split_ids(s, split_root)
        except Exception:
            split_manifest[s] = []

    seq_ids = split_manifest.get(split_name, load_split_ids(split_name, split_root))
    logger.info("split=%s: %d sequences", split_name, len(seq_ids))

    stats: Dict[str, Any] = {
        "split"           : split_name,
        "n_sequences"     : len(seq_ids),
        "n_phases"        : 0,
        "n_samples"       : 0,
        "bucket_dist"     : {},
        "fallback_seqs"   : 0,
        "contact_sample_count"  : 0,
    }

    all_phase_metadata : List[Dict]           = []
    all_conflict_logs  : Dict[str, List[dict]] = {}
    all_sm_logs        : Dict[str, List[dict]] = {}
    all_duration_logs  : Dict[str, List[dict]] = {}

    for seq_id in seq_ids:
        try:
            sequence = load_sequence_from_h5(seq_id, h5_path)
            sequence = compute_world_joints(sequence, body_model=body_model, device=fk_device)
        except Exception as e:
            logger.error("skip seq %s: %s", seq_id, e)
            continue

        mode_labels, sm = replay_and_label_modes(sequence, foot_joint_indices)

        # Collect SM entry/exit log from state machine
        sm_state = sm.get_state() if hasattr(sm, "get_state") else {}
        all_sm_logs[seq_id] = getattr(sm, "_event_log", [])

        ann_path = os.path.join(annotation_dir, seq_id) if annotation_dir else None
        phases = segment_interaction_phases(
            sequence, mode_labels,
            annotation_path=ann_path,
            cache_path=cache_path,
            keyword_map_path=keyword_map_path,
        )

        if not phases:
            stats["fallback_seqs"] += 1

        interaction_category = _action_name_from_seq_id(seq_id)
        for phase_meta in phases:
            stats["n_phases"] += 1
            bucket = phase_meta.get("duration_bucket", 0)
            stats["bucket_dist"][str(bucket)] = stats["bucket_dist"].get(str(bucket), 0) + 1

            phase_id = phase_meta.get("phase_id", f"{seq_id}_unknown")
            all_phase_metadata.append(phase_meta)

            # Duration resolver log stored in phase_meta
            dur_log = phase_meta.get("duration_resolve_log")
            if dur_log:
                all_duration_logs[phase_id] = dur_log if isinstance(dur_log, list) else [dur_log]

            for actor_id in ["P1", "P2"]:
                text_key   = "p1_text" if actor_id == "P1" else "p2_text"
                phase_text = phase_meta.get(text_key, "")

                try:
                    sample = build_phase_sample(
                        actor_id=actor_id,
                        phase_meta=phase_meta,
                        sequence=sequence,
                        foot_joint_indices=foot_joint_indices or [],
                        phase_text=phase_text,
                        phase_type=phase_meta.get("phase_type", "approach"),
                        interaction_category=interaction_category,
                    )

                    if sample.contact_frame_mask is not None and sample.contact_frame_mask.any():
                        stats["contact_sample_count"] += 1

                    # Collect conflict logs from phase_meta
                    cl_key = f"{phase_id}_{actor_id}"
                    conflict_log = phase_meta.get(f"conflict_log_{actor_id}", [])
                    if conflict_log:
                        all_conflict_logs[cl_key] = conflict_log

                    fname = os.path.join(
                        output_dir, split_name,
                        f"{sample.phase_id}_{actor_id}.npz",
                    )
                    _save_sample(sample, fname)
                    stats["n_samples"] += 1

                except Exception as e:
                    logger.error(
                        "build_phase_sample failed seq=%s phase=%s actor=%s: %s",
                        seq_id, phase_meta.get("phase_id"), actor_id, e,
                    )

    # Write dataset_stats.json
    if stats["n_samples"] > 0:
        stats["contact_ratio"] = stats["contact_sample_count"] / stats["n_samples"]
    else:
        stats["contact_ratio"] = 0.0

    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("dataset_stats written to %s", stats_path)

    # Write H5 metadata
    export_metadata_h5(
        output_dir=output_dir,
        split_name=split_name,
        phase_metadata=all_phase_metadata,
        conflict_logs=all_conflict_logs,
        sm_logs=all_sm_logs,
        duration_logs=all_duration_logs,
        split_manifest=split_manifest,
    )
