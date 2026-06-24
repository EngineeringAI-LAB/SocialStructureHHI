"""
hhi_types.py
-------------
纯数据类型定义：dataclass / TypedDict / Enum。
禁止依赖 PyTorch 模型实现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ── Enums ─────────────────────────────────────────────────────────────────────

class InteractionMode(str, Enum):
    NAVIGATION   = "NAVIGATION_MODE"
    INTERACTION  = "INTERACTION_MODE"


class ActorRole(str, Enum):
    PING = "P1"   # Actor / initiator;  data dim [0:3] in inter-x_regen.h5
    PONG = "P2"   # Reactor / responder; data dim [3:6] in inter-x_regen.h5


class PartnerSource(str, Enum):
    GT              = "gt"
    CACHE_GENERATED = "cache_generated"
    RUNTIME         = "runtime_generated"
    GT_FALLBACK     = "gt_fallback"


# ── Schema ────────────────────────────────────────────────────────────────────

NA = "[N/A]"   # canonical null token string used across all vocab fields


@dataclass
class Schema:
    """
    单个 Phase 的结构化动作指令。所有字段必须来自闭集词表；
    未知/不适用项统一填 NA = '[N/A]'。

    用途：内部中间表示（GT 接触检测 → 文本描述转换）。
    不再作为模型输入 token（已由自然语言 phase_text 取代）。
    """
    action        : str = NA
    actor_part    : str = NA
    target_part   : str = NA
    spatial       : str = NA
    contact       : str = NA
    orientation   : str = NA
    priority      : str = NA
    actor_id      : str = NA   # "P1" | "P2" | "[N/A]"

    def is_contact_schema(self) -> bool:
        return self.contact not in (NA, "no_contact")

    def is_na(self) -> bool:
        """全字段均为 NA 时视为空槽。"""
        return all(
            v in (NA, "no_contact")
            for v in (self.action, self.actor_part, self.target_part,
                      self.spatial, self.contact, self.orientation, self.priority)
        )


# ── Duration Resolver Record ──────────────────────────────────────────────────

@dataclass
class DurationResolveRecord:
    """记录 DurationBucketResolver 的单次决策日志。"""
    phase_id        : str
    raw_duration    : int
    resolved_bucket : int
    decision        : str            # "exact_match" | "phase_split" | "upward_snap" | "na_tail_fill"
    split_children  : List[str] = field(default_factory=list)   # child phase ids if split
    tail_fill_frames: int = 0        # [N/A] wait tail 帧数


# ── Phase Sample ──────────────────────────────────────────────────────────────

@dataclass
class PhaseSample:
    """
    单个 actor 在单个 Phase 上的训练样本。
    所有 motion 数据以 numpy ndarray 形式存储（转 tensor 在 Dataset.__getitem__ 完成）。

    Dimension conventions (all in Ego-centric local coordinates):
      self_history_local      : [W_PREV, D_motion]
      partner_history_local   : [W_PREV, D_motion]
      partner_rollout_local   : [W, D_motion]  -- Pong only; None for Ping
      target_motion_local     : [W, D_motion]
      contact_frame_mask      : [W]  bool
      seq_mask                : [W]  bool  -- True for real frames, False for [N/A] tail padding
      prefix_mask             : [W]  bool  -- True for overlap/inpaint prefix (read-only region)
      partner_time_ids        : [W_PREV]  for Ping; [W_PREV + W]  for Pong

    phase_text: natural language description of this actor's action in this phase.
      Example: "P1 hugs P2's shoulders with both arms, making firm contact."
      Encoded via HunyuanMotion's text encoder at training/inference time.
    """
    # ── Identifiers ───────────────────────────────────────────────────────────
    sequence_id      : str
    phase_id         : str
    actor_id         : str            # "P1" | "P2"
    phase_len_bucket : int            # one of DURATION_BUCKETS
    mode             : InteractionMode = InteractionMode.INTERACTION

    # ── Motion tensors (numpy, pre-egocentric) ────────────────────────────────
    self_history_local     : Optional[object] = None   # np.ndarray [W_PREV, D]
    partner_history_local  : Optional[object] = None   # np.ndarray [W_PREV, D]

    # Partner rollout — 两个字段分别对应时序 Ping-Pong 的两种用途：
    #   partner_prev_local: partner 上一个 phase 的 GT（Phase 0: partner_f0×W）
    #                       P1 (Ping) 使用
    #   partner_curr_local: partner 当前 phase 的 GT（Phase 0: partner_f0×W）
    #                       P2 (Pong) 使用（teacher forcing）
    # Phase 0 两字段内容相同。训练/推理代码按 actor_id 或 is_phase0 选择字段。
    partner_prev_local     : Optional[object] = None   # np.ndarray [W_prev, D]
    partner_prev_seq_mask  : Optional[object] = None   # np.ndarray [W_prev] bool
    partner_curr_local     : Optional[object] = None   # np.ndarray [W, D]
    partner_curr_seq_mask  : Optional[object] = None   # np.ndarray [W] bool

    target_motion_local    : Optional[object] = None   # np.ndarray [W, D]

    # ── Global initial frames（phase 0 第 0 帧，in current phase ego 系）─────
    initial_frame          : Optional[object] = None   # np.ndarray [D]  self phase0 f0
    partner_initial_frame  : Optional[object] = None   # np.ndarray [D]  partner phase0 f0

    # ── Partner time encoding（W_PREV history + rollout）────────────────────
    partner_prev_time_ids  : Optional[object] = None   # np.ndarray int32 [W_PREV + W_prev]
    partner_curr_time_ids  : Optional[object] = None   # np.ndarray int32 [W_PREV + W]

    # ── Phase type & text ─────────────────────────────────────────────────────
    phase_type             : str = "approach"   # closed-set semantic label (approach/reach/contact_hold/...)
    interaction_category   : str = ""           # Inter-X action name (e.g. "Hug")
    phase_text             : str = ""           # natural language description for this actor's action
    global_text            : str = ""           # sequence-level description (from dataset annotation)

    # ── Contact ───────────────────────────────────────────────────────────────
    contact_frame_mask : Optional[object] = None   # np.ndarray [W] bool

    # ── Sequence masks ────────────────────────────────────────────────────────
    seq_mask    : Optional[object] = None   # np.ndarray [W] bool
    prefix_mask : Optional[object] = None   # np.ndarray [W] bool

    # ── Metadata (not used in training loss; kept for debug / evaluation) ─────
    t_start          : int = 0
    t_end            : int = 0
    partner_source   : PartnerSource = PartnerSource.GT
    is_split_child   : bool = False
    split_parent_id  : Optional[str] = None
    split_index      : int = 0        # 0-based index among sibling splits


# ── State Machine State ───────────────────────────────────────────────────────

@dataclass
class StateMachineState:
    """
    运行时迟滞状态机的完整状态快照。
    同时用于 GT 回放打标和推理期模式切换。
    """
    mode                    : InteractionMode = InteractionMode.NAVIGATION
    frame_idx               : int = 0
    contact_free_frames     : int = 0      # 当前 INTERACTION_MODE 下无接触连续帧计数
    cumulative_drift_m      : float = 0.0  # 相对 planner anchor 的累计平面漂移（米）

    # 全局大盘：P1/P2 当前世界坐标 root 位置
    p1_root_world           : Optional[object] = None   # np.ndarray [3]
    p2_root_world           : Optional[object] = None   # np.ndarray [3]

    # 全量关节坐标（世界系），用于 grounding 和评估
    p1_joints_world         : Optional[object] = None   # np.ndarray [J, 3]
    p2_joints_world         : Optional[object] = None   # np.ndarray [J, 3]

    # 历史 mode 列表（逐帧），用于 debug / failure replay
    mode_history            : List[InteractionMode] = field(default_factory=list)

    # 事件日志
    entry_events            : List[dict] = field(default_factory=list)
    exit_events             : List[dict] = field(default_factory=list)


# ── Planner Output ────────────────────────────────────────────────────────────

@dataclass
class SpatialTarget:
    target_root_distance : float = 1.0
    target_bearing_deg   : float = 0.0


@dataclass
class PhasePlan:
    phase_id        : str
    mode            : InteractionMode
    executor_actor  : str            # "P1" | "P2"
    raw_duration    : int            # planner 原始建议帧数（保留，不修改）
    duration_bucket : int            # normalization 后生效值（可回写）
    phase_type           : str = "approach"   # closed-set semantic label; drives duration rule table
    tempo                : str = "medium"     # "slow" | "medium" | "fast"; LLM-chosen rhythm
    interaction_category : str = ""           # Inter-X action name (e.g. "Hug"), used for fine-grained duration lookup
    spatial_target  : SpatialTarget  = field(default_factory=SpatialTarget)
    p1_text         : str = ""       # natural language description of P1's action in this phase
    p2_text         : str = ""       # natural language description of P2's action in this phase
    has_contact     : bool = False   # True if either actor has physical contact in this phase


@dataclass
class PlannerOutput:
    sequence_goal   : str
    phases          : List[PhasePlan] = field(default_factory=list)
