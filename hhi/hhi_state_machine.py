"""
hhi_state_machine.py
---------------------
迟滞门控状态机、navigation/interaction mode 切换、
局部到全局状态推进、Y-axis grounding。

同一套实现同时用于：
  - GT 数据回放打标
  - 推理期模式切换
  - debug / failure replay

核心类：
  HysteresisStateMachine
  SimulationState（对外暴露的全局大盘快照）

禁止：直接读 dataset 文件；模型/PyTorch 代码。
"""
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .hhi_constants import (
    AIRBORNE_THRESH,
    CONTACT_DIST_THRESH,
    ENTRY_DIST,
    EXIT_CONTACT_FREE_FRAMES,
    EXIT_DIST,
    FOOT_JOINT_NAMES,
    GROUND_Y,
    JUMP_HOLD_FRAMES,
    OVERLAP_N,
)
from .hhi_types import InteractionMode, StateMachineState

logger = logging.getLogger(__name__)


# ── SimulationState ───────────────────────────────────────────────────────────

@dataclass
class SimulationState:
    """
    全局大盘状态，供 compose_local_to_global 和 debug 使用。
    joints_world: np.ndarray [J, 3] 世界系关节坐标（Y-grounded）
    root_world  : np.ndarray [3]    世界系 root 位置
    root_rot_world: np.ndarray [4]  世界系 root 旋转（四元数 xyzw）
    """
    p1_joints_world    : Optional[np.ndarray] = None
    p1_root_world      : Optional[np.ndarray] = None
    p1_root_rot_world  : Optional[np.ndarray] = None

    p2_joints_world    : Optional[np.ndarray] = None
    p2_root_world      : Optional[np.ndarray] = None
    p2_root_rot_world  : Optional[np.ndarray] = None

    frame_idx          : int = 0


# ── apply_y_grounding ─────────────────────────────────────────────────────────

def apply_y_grounding(
    joints            : np.ndarray,             # [J, 3] world-space joint positions
    foot_joint_indices: List[int],
    airborne_thresh   : float = AIRBORNE_THRESH,
    jump_hold_frames  : int   = JUMP_HOLD_FRAMES,
    prev_airborne_count: int  = 0,
    ground_y          : float = GROUND_Y,
) -> Tuple[np.ndarray, int]:
    """
    对单帧全身关节坐标执行 Y-grounding。

    算法：
      1. 取 foot_joint_indices 中 Y 坐标最小值 y_min_foot
      2. sink = max(0, GROUND_Y - y_min_foot)
      3. 若该帧腾空（y_min_foot > AIRBORNE_THRESH 且 root 向上或连续 airborne），
         则不做 grounding 修正
      4. 非腾空帧：全身关节 Y += sink（仅向上，不向下）
      5. 返回修正后的 joints 和更新后的 airborne_count

    参数：
      prev_airborne_count : 前几帧连续腾空计数（0 表示上一帧未腾空）
    """
    joints = joints.copy()

    y_foot = joints[foot_joint_indices, 1]   # [n_foot]
    y_min  = float(y_foot.min())

    # 腾空判定
    is_airborne = (
        y_min > airborne_thresh
        or prev_airborne_count >= jump_hold_frames
    )

    if is_airborne:
        airborne_count = prev_airborne_count + 1
        return joints, airborne_count

    # 非腾空：向上补偿
    sink = max(0.0, ground_y - y_min)
    if sink > 0.0:
        joints[:, 1] += sink

    return joints, 0


def apply_y_grounding_sequence(
    joints_seq        : np.ndarray,    # [T, J, 3]
    foot_joint_indices: Optional[List[int]],
    airborne_thresh   : float = AIRBORNE_THRESH,
    jump_hold_frames  : int   = JUMP_HOLD_FRAMES,
    ground_y          : float = GROUND_Y,
) -> np.ndarray:
    """对整段序列逐帧执行 apply_y_grounding。"""
    # Default foot indices if not provided: [LeftAnkle, RightAnkle, LeftToe, RightToe]
    if foot_joint_indices is None:
        foot_joint_indices = [7, 8, 10, 11]
    
    T = joints_seq.shape[0]
    result       = joints_seq.copy()
    airborne_cnt = 0
    for t in range(T):
        result[t], airborne_cnt = apply_y_grounding(
            result[t], foot_joint_indices,
            airborne_thresh, jump_hold_frames, airborne_cnt, ground_y,
        )
    return result


# ── Ego-centric coordinate transform ─────────────────────────────────────────

def _yaw_rotation_matrix(yaw_rad: float) -> np.ndarray:
    """返回绕 Y 轴旋转 yaw_rad 的 3x3 矩阵（右手系）。"""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float32)


def _extract_root_yaw(root_rot: np.ndarray) -> float:
    """
    从 root 旋转（四元数 xyzw 或旋转矩阵 [3,3]）提取 yaw 角（绕 Y 轴）。
    此处假设输入为四元数 [qx, qy, qz, qw]。
    """
    if root_rot.shape == (4,):
        qx, qy, qz, qw = root_rot
        yaw = np.arctan2(2 * (qw * qy + qz * qx), 1 - 2 * (qy * qy + qz * qz))
        return float(yaw)
    elif root_rot.shape == (3, 3):
        # R[0,2] = sin(yaw), R[2,2] = cos(yaw) for pure yaw
        return float(np.arctan2(root_rot[0, 2], root_rot[2, 2]))
    raise ValueError(f"Unsupported root_rot shape: {root_rot.shape}")


def build_egocentric_transform(
    root_pos: np.ndarray,   # [3]
    root_rot: np.ndarray,   # [4] quat xyzw  or  [3,3] rot matrix
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建 Ego-centric 变换：
      origin  = root_pos (xz) with y preserved (grounded separately)
      yaw     = root_rot yaw angle

    返回：
      origin  : np.ndarray [3]   translation offset
      R_inv   : np.ndarray [3,3] inverse yaw rotation matrix
    """
    yaw    = _extract_root_yaw(root_rot)
    R      = _yaw_rotation_matrix(yaw)
    R_inv  = R.T   # orthogonal → inverse = transpose
    origin = root_pos.copy()
    return origin, R_inv


def apply_egocentric_transform(
    joints   : np.ndarray,   # [..., 3]
    origin   : np.ndarray,   # [3]
    R_inv    : np.ndarray,   # [3, 3]
) -> np.ndarray:
    """将世界系关节坐标转换到局部 Ego-centric 坐标。"""
    shifted = joints - origin   # subtract reference point
    return (R_inv @ shifted.reshape(-1, 3).T).T.reshape(joints.shape)


def apply_egocentric_transform_inv(
    joints_local: np.ndarray,  # [..., 3]
    origin       : np.ndarray,  # [3]
    R            : np.ndarray,  # [3, 3]  forward rotation
) -> np.ndarray:
    """将 Ego-centric 局部关节坐标变换回世界系。"""
    rotated = (R @ joints_local.reshape(-1, 3).T).T.reshape(joints_local.shape)
    return rotated + origin


# ── HysteresisStateMachine ────────────────────────────────────────────────────

class HysteresisStateMachine:
    """
    双阈值迟滞状态机。

    同一实例可以：
      (a) replay_sequence()     -- GT 回放打标，输出逐帧 mode 标签
      (b) update_mode() / step  -- 推理期逐帧调用
    """

    def __init__(
        self,
        entry_dist              : float = ENTRY_DIST,
        exit_dist               : float = EXIT_DIST,
        exit_contact_free_frames: int   = EXIT_CONTACT_FREE_FRAMES,
        foot_joint_indices      : Optional[List[int]] = None,
    ) -> None:
        self.entry_dist               = entry_dist
        self.exit_dist                = exit_dist
        self.exit_contact_free_frames = exit_contact_free_frames
        self.foot_joint_indices       = foot_joint_indices or []

        self._reset()

    def _reset(self) -> None:
        self.mode                = InteractionMode.NAVIGATION
        self.contact_free_frames = 0
        self._airborne_p1        = 0
        self._airborne_p2        = 0
        self.entry_events: List[dict] = []
        self.exit_events : List[dict] = []

    # ── Mode Update ───────────────────────────────────────────────────────────

    def update_mode(
        self,
        frame_idx      : int,
        root_dist      : float,
        has_contact    : bool = False,
    ) -> InteractionMode:
        """
        根据双人 root 距离和接触状态更新模式。

        规则：
          1. NAVIGATION + dist < entry_dist  → INTERACTION
          2. INTERACTION + dist > exit_dist + 连续 exit_contact_free_frames 帧无接触 → NAVIGATION
          3. 灰区保持上一状态
        """
        prev_mode = self.mode

        if self.mode == InteractionMode.NAVIGATION:
            if root_dist < self.entry_dist:
                self.mode = InteractionMode.INTERACTION
                self.contact_free_frames = 0
                self.entry_events.append({
                    "frame"    : frame_idx,
                    "root_dist": root_dist,
                })
                logger.debug("frame %d: NAVIGATION → INTERACTION (dist=%.3fm)", frame_idx, root_dist)

        elif self.mode == InteractionMode.INTERACTION:
            if has_contact:
                self.contact_free_frames = 0
            else:
                self.contact_free_frames += 1

            if (root_dist > self.exit_dist
                    and self.contact_free_frames >= self.exit_contact_free_frames):
                self.mode = InteractionMode.NAVIGATION
                self.contact_free_frames = 0
                self.exit_events.append({
                    "frame"              : frame_idx,
                    "root_dist"          : root_dist,
                    "contact_free_frames": self.contact_free_frames,
                })
                logger.debug("frame %d: INTERACTION → NAVIGATION (dist=%.3fm)", frame_idx, root_dist)

        return self.mode

    # ── Replay / Label ────────────────────────────────────────────────────────

    def replay_sequence(
        self,
        p1_roots   : np.ndarray,          # [T, 3] P1 root positions (world)
        p2_roots   : np.ndarray,          # [T, 3] P2 root positions (world)
        contact_flags: Optional[np.ndarray] = None,  # [T] bool
    ) -> np.ndarray:
        """
        对整条序列做前向回放，返回逐帧 mode 标签数组。

        返回：
          mode_labels : np.ndarray [T]  dtype=object (InteractionMode)
        """
        T = p1_roots.shape[0]
        if contact_flags is None:
            contact_flags = np.zeros(T, dtype=bool)

        self._reset()
        labels = np.empty(T, dtype=object)
        for t in range(T):
            dist      = float(np.linalg.norm(p1_roots[t, [0, 2]] - p2_roots[t, [0, 2]]))
            labels[t] = self.update_mode(t, dist, bool(contact_flags[t]))
        return labels

    # ── compose_local_to_global ───────────────────────────────────────────────

    def compose_local_to_global(
        self,
        local_joints  : np.ndarray,   # [W, J, 3]  ego-centric 输出
        ref_origin    : np.ndarray,   # [3]         局部参考点（世界系）
        ref_R         : np.ndarray,   # [3, 3]      局部坐标系前向旋转矩阵（非逆）
        prev_world_tail: Optional[np.ndarray] = None,  # [J, 3] 上一段最后帧（用于连续性）
    ) -> np.ndarray:
        """
        将 ego-centric 输出序列累加到世界系。

        做法：直接用参考帧刚体变换将局部坐标映射回世界坐标，
        不做累积积分漂移（Ego-centric 假设下 ref_origin 每段重新锚定）。

        返回：world_joints [W, J, 3]
        """
        W, J, _ = local_joints.shape
        world_joints = apply_egocentric_transform_inv(
            local_joints.reshape(-1, 3), ref_origin, ref_R,
        ).reshape(W, J, 3)
        return world_joints

    # ── apply_y_grounding (世界系 in-place) ──────────────────────────────────

    def ground_sequence(
        self,
        joints_seq : np.ndarray,   # [T, J, 3]
        actor_id   : str,          # "P1" | "P2"  -- 只用于日志
    ) -> np.ndarray:
        """对整段序列应用 Y-grounding，更新内部 airborne 计数。"""
        T = joints_seq.shape[0]
        result = joints_seq.copy()
        airborne_cnt = (
            self._airborne_p1 if actor_id == "P1" else self._airborne_p2
        )
        for t in range(T):
            result[t], airborne_cnt = apply_y_grounding(
                result[t],
                self.foot_joint_indices,
                prev_airborne_count=airborne_cnt,
            )
        if actor_id == "P1":
            self._airborne_p1 = airborne_cnt
        else:
            self._airborne_p2 = airborne_cnt
        return result

    # ── Contact detection (world space) ──────────────────────────────────────

    @staticmethod
    def detect_contact(
        p1_joints        : np.ndarray,           # [J, 3]
        p2_joints        : np.ndarray,           # [J, 3]
        contact_pairs_idx: List[Tuple[int, int]], # [(p1_joint_idx, p2_joint_idx)]
        threshold        : float = CONTACT_DIST_THRESH,
    ) -> bool:
        """若任意一对关节距离 < threshold 则视为接触。"""
        for i, j in contact_pairs_idx:
            d = np.linalg.norm(p1_joints[i] - p2_joints[j])
            if d < threshold:
                return True
        return False

    # ── State snapshot ────────────────────────────────────────────────────────

    def get_state(self) -> StateMachineState:
        return StateMachineState(
            mode                = self.mode,
            contact_free_frames = self.contact_free_frames,
            entry_events        = list(self.entry_events),
            exit_events         = list(self.exit_events),
        )
