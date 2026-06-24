"""
hhi_duration_buckets.py
------------------------
Duration bucket 合法化、phase split、[N/A] tail fill。

核心函数：
  resolve_duration()        -- 单 phase 时长 → 合法 bucket
  split_phase_if_needed()   -- 必要时做语义 phase split
  bucketize_phase_sequence()-- 批量处理 phase 列表并回写 duration_bucket

禁止：模型代码；直接读取 dataset 原始文件。
"""
from __future__ import annotations

import bisect
import logging
from typing import Any, Dict, List, Optional, Tuple

from .hhi_constants import DURATION_BUCKETS, OVERLAP_N
from .hhi_types import DurationResolveRecord

logger = logging.getLogger(__name__)

# ── 内部工具 ──────────────────────────────────────────────────────────────────

def _nearest_bucket(raw_len: int) -> int:
    """返回 ≥ raw_len 的最小合法 bucket；若超过最大桶则取最大桶。"""
    for b in DURATION_BUCKETS:
        if b >= raw_len:
            return b
    return DURATION_BUCKETS[-1]


def _exact_match(raw_len: int) -> Optional[int]:
    if raw_len in DURATION_BUCKETS:
        return raw_len
    return None


# ── 主决策函数 ────────────────────────────────────────────────────────────────

def resolve_duration(
    raw_len       : int,
    phase_id      : str,
    has_contact   : bool   = False,
    is_splittable : bool   = False,
    split_hint    : Optional[Tuple[int, int]] = None,
) -> DurationResolveRecord:
    """
    将 raw_len 映射到合法 bucket，返回 DurationResolveRecord。

    决策优先级（严格按此顺序）：
      1. 精确命中 bucket → "exact_match"
      2. 可语义拆分（is_splittable=True）且存在合法子桶组合 → "phase_split"
      3. 接触段（has_contact=True）或不可拆 → 向上吸附最小合法桶 → "upward_snap"
         若存在 tail，追加 [N/A] wait tail → "na_tail_fill"（caller 负责填充数据）

    参数：
      split_hint : 若提供 (a, b) 表示建议将 phase 拆为 a + b 帧；否则自动搜索。
    """
    # 1. 精确命中
    if _exact_match(raw_len) is not None:
        return DurationResolveRecord(
            phase_id        = phase_id,
            raw_duration    = raw_len,
            resolved_bucket = raw_len,
            decision        = "exact_match",
        )

    # 2. 语义拆分（仅当 is_splittable=True）
    if is_splittable:
        split = _find_split(raw_len, split_hint)
        if split is not None:
            a, b = split
            return DurationResolveRecord(
                phase_id        = phase_id,
                raw_duration    = raw_len,
                resolved_bucket = a,   # 父 phase 取第一段 bucket；子 phase 另建 record
                decision        = "phase_split",
                split_children  = [f"{phase_id}_split0", f"{phase_id}_split1"],
                tail_fill_frames= 0,
            )

    # 3. 向上吸附
    bucket = _nearest_bucket(raw_len)
    tail   = bucket - raw_len
    decision = "na_tail_fill" if tail > 0 else "upward_snap"
    if tail > 0:
        logger.debug(
            "phase %s: raw=%d → bucket=%d, [N/A] tail=%d frames",
            phase_id, raw_len, bucket, tail,
        )
    return DurationResolveRecord(
        phase_id        = phase_id,
        raw_duration    = raw_len,
        resolved_bucket = bucket,
        decision        = decision,
        tail_fill_frames= tail,
    )


def _find_split(
    raw_len   : int,
    hint      : Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[int, int]]:
    """
    搜索合法的 (a, b) bucket 组合使得 a + b == raw_len，考虑 OVERLAP_N。

    注意：split 后两段共享 OVERLAP_N 帧 overlap，因此实际原始帧数需满足：
      a + b - OVERLAP_N == raw_len
      即搜索 a + b = raw_len + OVERLAP_N

    返回第一个找到的 (a, b)；若无合法组合返回 None。
    """
    if hint is not None:
        a, b = hint
        if a in DURATION_BUCKETS and b in DURATION_BUCKETS:
            if a + b - OVERLAP_N == raw_len or a + b == raw_len:
                return (a, b)

    target = raw_len + OVERLAP_N
    for a in DURATION_BUCKETS:
        b = target - a
        if b in DURATION_BUCKETS and a > 0 and b > 0:
            return (a, b)
    # 不考虑 overlap 的简单分割 fallback
    for a in DURATION_BUCKETS:
        b = raw_len - a
        if b in DURATION_BUCKETS and a > 0 and b > 0:
            return (a, b)
    return None


# ── Phase Split ───────────────────────────────────────────────────────────────

def split_phase_if_needed(
    phase_meta      : Dict[str, Any],
    contact_segments: Optional[List[Tuple[int, int]]] = None,
) -> List[Dict[str, Any]]:
    """
    若 phase_meta['raw_duration'] 落在桶间且可语义拆分，则拆成两个子 phase_meta。

    参数：
      phase_meta       : 包含至少 phase_id / raw_duration / is_splittable / t_start 字段
      contact_segments : [(start_frame, end_frame)] 该 phase 内的接触段；
                         若接触段跨越拆分点，则禁止在此处拆分。

    返回：phase_meta 列表（长度 1 = 不拆；长度 2 = 拆成两段）。
    每个子 phase_meta 新增字段：
      is_split_child, split_parent_id, split_index, duration_bucket
    """
    raw_len  = phase_meta["raw_duration"]
    phase_id = phase_meta["phase_id"]

    if _exact_match(raw_len) is not None:
        phase_meta["duration_bucket"] = raw_len
        return [phase_meta]

    if not phase_meta.get("is_splittable", False):
        rec = resolve_duration(raw_len, phase_id, has_contact=True)
        phase_meta["duration_bucket"] = rec.resolved_bucket
        phase_meta["tail_fill_frames"] = rec.tail_fill_frames
        return [phase_meta]

    split = _find_split(raw_len)
    if split is None:
        rec = resolve_duration(raw_len, phase_id, is_splittable=True)
        phase_meta["duration_bucket"] = rec.resolved_bucket
        phase_meta["tail_fill_frames"] = rec.tail_fill_frames
        return [phase_meta]

    a, b    = split
    t_start = phase_meta.get("t_start", 0)
    split_point = t_start + a
    # 判断是否使用了带 overlap 的分割：a+b-OVERLAP_N==raw_len
    # fallback 分支（a+b==raw_len）不应减去 OVERLAP_N，否则 split1 末端会短 OVERLAP_N 帧
    use_overlap = (a + b - OVERLAP_N == raw_len)

    # 检查拆分点是否在接触段内（禁止跨接触段拆分）
    if contact_segments:
        for cs, ce in contact_segments:
            if cs < split_point < ce:
                logger.warning(
                    "phase %s: 拆分点 %d 在接触段 [%d, %d] 内，改为向上吸附",
                    phase_id, split_point, cs, ce,
                )
                rec = resolve_duration(raw_len, phase_id, has_contact=True)
                phase_meta["duration_bucket"] = rec.resolved_bucket
                phase_meta["tail_fill_frames"] = rec.tail_fill_frames
                return [phase_meta]

    # 执行拆分
    raw_len  = phase_meta["raw_duration"]
    t_start  = phase_meta.get("t_start", 0)
    t_end    = phase_meta.get("t_end", t_start + raw_len)
    
    child0 = {
        **phase_meta,
        "phase_id"       : f"{phase_id}_split0",
        "raw_duration"   : a,
        "duration_bucket": a,
        "t_start"        : t_start,
        "t_end"          : t_start + a,
        "is_split_child" : True,
        "split_parent_id": phase_id,
        "split_index"    : 0,
        "tail_fill_frames": 0,
    }
    # child1 的起点要考虑 overlap
    overlap_start = (t_start + a - OVERLAP_N) if use_overlap else (t_start + a)
    child1_len = t_end - overlap_start  # 实际数据长度
    
    # child1_len 需要向上吸附到合法 bucket
    child1_bucket = _nearest_bucket(child1_len)
    
    child1 = {
        **phase_meta,
        "phase_id"       : f"{phase_id}_split1",
        "raw_duration"   : child1_len,
        "duration_bucket": child1_bucket,
        "t_start"        : overlap_start,
        "t_end"          : t_end,
        "is_split_child" : True,
        "split_parent_id": phase_id,
        "split_index"    : 1,
        "tail_fill_frames": child1_bucket - child1_len,
    }
    logger.info(
        "phase %s split → [%s, %s] (overlap=%d)",
        phase_id, child0["phase_id"], child1["phase_id"], OVERLAP_N,
    )
    return [child0, child1]


# ── 批量处理 ──────────────────────────────────────────────────────────────────

def bucketize_phase_sequence(
    phases: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[DurationResolveRecord]]:
    """
    对整条 sequence 的 phase 列表做批量 bucket 合法化。

    - 触发 split 时将新子 phase 插入原位置
    - 所有决策都回写到 phase_meta['duration_bucket']
    - 返回更新后的 phases 列表 + 决策日志列表
    """
    result  : List[Dict[str, Any]]        = []
    records : List[DurationResolveRecord] = []

    for phase_meta in phases:
        raw_len   = phase_meta["raw_duration"]
        phase_id  = phase_meta["phase_id"]
        has_contact = phase_meta.get("has_contact", False)

        rec = resolve_duration(
            raw_len,
            phase_id,
            has_contact   = has_contact,
            is_splittable = phase_meta.get("is_splittable", False),
        )
        records.append(rec)

        if rec.decision == "phase_split":
            children = split_phase_if_needed(phase_meta)
            result.extend(children)
        else:
            phase_meta["duration_bucket"]  = rec.resolved_bucket
            phase_meta["tail_fill_frames"] = rec.tail_fill_frames
            result.append(phase_meta)

    return result, records
