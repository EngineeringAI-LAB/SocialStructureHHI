"""
hhi_schema.py
--------------
Schema 词表加载、序列化、conflict resolution、slot padding。

禁止：直接读取 dataset 原始文件；放任何模型/PyTorch 代码。
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .hhi_constants import MAX_SLOTS
from dataclasses import dataclass
from typing import List as _List
from .hhi_types import NA, Schema


# SerializedSchema kept here for backward compat with pack_schema_list / evaluator.
@dataclass
class SerializedSchema:
    action_id       : int = 0
    actor_part_id   : int = 0
    target_part_id  : int = 0
    spatial_id      : int = 0
    contact_id      : int = 0
    orientation_id  : int = 0
    priority_id     : int = 0

    @property
    def as_list(self) -> _List[int]:
        return [
            self.action_id, self.actor_part_id, self.target_part_id,
            self.spatial_id, self.contact_id, self.orientation_id, self.priority_id,
        ]

    @classmethod
    def num_fields(cls) -> int:
        return 7

logger = logging.getLogger(__name__)

# ── 词表文件路径 ──────────────────────────────────────────────────────────────
_VOCAB_DIR = os.path.join(os.path.dirname(__file__), "vocabs")

_VOCAB_FILES: Dict[str, str] = {
    "action"     : "action_vocab.json",
    "body_part"  : "body_part_vocab.json",
    "spatial"    : "spatial_vocab.json",
    "contact"    : "contact_vocab.json",
    "orientation": "orientation_vocab.json",
    "priority"   : "priority_vocab.json",
}

# ── 全局词表缓存 ──────────────────────────────────────────────────────────────
_VOCABS: Optional[Dict[str, Dict[str, int]]] = None


def load_schema_vocabs(vocab_dir: str = _VOCAB_DIR) -> Dict[str, Dict[str, int]]:
    """
    加载所有闭集词表；结果缓存在模块级变量中，只读一次磁盘。
    返回 {vocab_name -> {token_str -> int_id}}。
    """
    global _VOCABS
    if _VOCABS is not None:
        return _VOCABS

    vocabs: Dict[str, Dict[str, int]] = {}
    for name, fname in _VOCAB_FILES.items():
        path = os.path.join(vocab_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            vocabs[name] = json.load(f)
        assert NA in vocabs[name], f"词表 {fname} 缺少 '{NA}' 条目"
    _VOCABS = vocabs
    return _VOCABS


def get_vocab_size(vocab_name: str) -> int:
    vocabs = load_schema_vocabs()
    return len(vocabs[vocab_name])


# ── 序列化 ────────────────────────────────────────────────────────────────────

def _lookup(vocab: Dict[str, int], token: str) -> int:
    """词表 lookup；未登录项返回 NA id 并记警告。"""
    if token in vocab:
        return vocab[token]
    logger.warning("未登录 token '%s'，回退到 [N/A]", token)
    return vocab[NA]


def serialize_schema(schema: Schema) -> SerializedSchema:
    """将单个 Schema 编码为定长 int id 向量。"""
    vocabs = load_schema_vocabs()
    return SerializedSchema(
        action_id      = _lookup(vocabs["action"],      schema.action),
        actor_part_id  = _lookup(vocabs["body_part"],   schema.actor_part),
        target_part_id = _lookup(vocabs["body_part"],   schema.target_part),
        spatial_id     = _lookup(vocabs["spatial"],     schema.spatial),
        contact_id     = _lookup(vocabs["contact"],     schema.contact),
        orientation_id = _lookup(vocabs["orientation"], schema.orientation),
        priority_id    = _lookup(vocabs["priority"],    schema.priority),
    )


def pack_schema_list(
    schema_list : List[Schema],
    max_slots   : int = MAX_SLOTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 schema_list 打包成定长张量。

    返回：
      schema_tokens  : np.ndarray  [max_slots, F_schema]  int32
      schema_val_mask: np.ndarray  [max_slots]             bool
        True  = 真实 schema 槽
        False = 填充的 [N/A] 槽
    """
    F = SerializedSchema.num_fields()
    tokens   = np.zeros((max_slots, F), dtype=np.int32)
    val_mask = np.zeros(max_slots,      dtype=bool)

    for i, schema in enumerate(schema_list[:max_slots]):
        serialized      = serialize_schema(schema)
        tokens[i]       = serialized.as_list
        val_mask[i]     = True

    # 剩余槽用全 [N/A] 补齐；tokens 已初始化为 0 即 NA id，无需额外操作
    return tokens, val_mask


# ── NA Schema 工厂 ────────────────────────────────────────────────────────────

def make_na_schema(actor_id: str = NA) -> Schema:
    return Schema(actor_id=actor_id)


def make_wait_schema(actor_id: str = NA) -> Schema:
    return Schema(
        action="wait", contact="no_contact", orientation="face_to_face",
        actor_id=actor_id,
    )


def make_fallback_schema(actor_id: str = NA) -> Schema:
    """用于 fallback phase：wait + face_to_face + no_contact + safety priority。"""
    return Schema(
        action="wait", contact="no_contact", orientation="face_to_face",
        priority="safety", actor_id=actor_id,
    )


# ── Conflict Resolution ───────────────────────────────────────────────────────

def _schema_priority_rank(schema: Schema) -> int:
    """数值越大，优先级越高。"""
    if schema.is_contact_schema():
        return 2    # contact 最高
    if schema.action not in (NA, "wait"):
        return 1    # gesture / spatial 次之
    return 0        # wait / [N/A] 最低


def resolve_schema_conflicts(
    schema_list: List[Schema],
    actor_id   : str = NA,
) -> Tuple[List[Schema], List[dict]]:
    """
    对同一 Phase 内的多槽 schema list 做冲突裁决。

    规则：
      1. Contact 优先于 gesture/spatial
      2. 仅当同等级且作用于同一 actor_part 时视为物理互斥
      3. 互斥时保留先到的主 schema，丢弃后者并记录 debug 日志
      4. 非同 body-part 的动作允许并存

    返回：
      resolved    : 裁决后的 schema list（最多 MAX_SLOTS 个）
      conflict_log: 每次丢弃事件的详细记录
    """
    resolved    : List[Schema] = []
    conflict_log: List[dict]   = []

    # 按优先级降序排列，保证高优先级 schema 先占位
    sorted_list = sorted(schema_list, key=_schema_priority_rank, reverse=True)

    occupied: Dict[str, int] = {}  # actor_part -> priority_rank of occupying schema

    for schema in sorted_list:
        rank = _schema_priority_rank(schema)
        part = schema.actor_part

        if part != NA and part in occupied:
            existing_rank = occupied[part]
            if existing_rank >= rank:
                # 互斥冲突：丢弃当前 schema
                conflict_log.append({
                    "dropped_action"  : schema.action,
                    "dropped_part"    : schema.actor_part,
                    "dropped_rank"    : rank,
                    "kept_rank"       : existing_rank,
                    "actor_id"        : actor_id,
                    "reason"          : "same_part_lower_or_equal_priority",
                })
                continue

        resolved.append(schema)
        if part != NA:
            occupied[part] = rank

    return resolved[:MAX_SLOTS], conflict_log


# ── Inter-X 文本解析 ──────────────────────────────────────────────────────────

def _load_cache(cache_path: str) -> Dict[str, dict]:
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_keyword_map(
    keyword_map_path: Optional[str] = None,
) -> Dict[str, dict]:
    if keyword_map_path is None:
        keyword_map_path = os.path.join(_VOCAB_DIR, "keyword_schema_map.json")
    if not os.path.exists(keyword_map_path):
        return {}
    with open(keyword_map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dict_to_schema(d: dict, actor_id: str = NA) -> Schema:
    return Schema(
        action      = d.get("action",      NA),
        actor_part  = d.get("actor_part",  NA),
        target_part = d.get("target_part", NA),
        spatial     = d.get("spatial",     NA),
        contact     = d.get("contact",     NA),
        orientation = d.get("orientation", NA),
        priority    = d.get("priority",    NA),
        actor_id    = actor_id,
    )


def parse_schema_from_text(
    text           : str,
    actor_id       : str = NA,
    cache_path     : Optional[str] = None,
    keyword_map_path: Optional[str] = None,
) -> Tuple[Schema, str]:
    """
    从自由文本动作描述中提取 Schema。

    解析路径（优先级降序）：
      1. 离线 LLM 缓存查表（schema_annotation_cache.json）
      2. 关键词 fallback 映射表（keyword_schema_map.json）
      3. 完全未命中 → 返回全 [N/A] Schema

    返回：
      schema : Schema
      source : "cache" | "keyword_<kw>" | "fallback"
    """
    text_lower = text.strip().lower()

    # Phase 1: 离线 LLM 缓存
    if cache_path is not None:
        cache = _load_cache(cache_path)
        if text in cache:
            return _dict_to_schema(cache[text], actor_id), "cache"
        if text_lower in cache:
            return _dict_to_schema(cache[text_lower], actor_id), "cache"

    # Phase 2: 关键词匹配
    kw_map = _load_keyword_map(keyword_map_path)
    for keyword, schema_dict in kw_map.items():
        if keyword in text_lower:
            return _dict_to_schema(schema_dict, actor_id), f"keyword_{keyword}"

    # Phase 3: 完全未命中
    logger.warning("schema 解析完全未命中: '%s'", text)
    return make_na_schema(actor_id), "fallback"


# ── Actor Alignment 断言 ──────────────────────────────────────────────────────

def assert_actor_alignment(schema: Schema, sample_actor_id: str) -> None:
    """
    断言 schema.actor_id 与 sample_actor_id 一致。
    在 build_phase_sample() 写入样本前调用，防止 schema 归属混乱。
    """
    if schema.actor_id not in (NA, sample_actor_id):
        raise AssertionError(
            f"Schema actor_id='{schema.actor_id}' != sample actor_id='{sample_actor_id}'. "
            "Schema 必须与当前样本执行者对齐。"
        )


def assert_skeleton_alignment(
    schema          : "Schema",
    contact_pairs   : list,
    sample_actor_id : str,
) -> None:
    """
    断言 schema.actor_part 与 contact_pairs 第一项 body part 保持一致：

    规则：
      1. 若 schema.actor_part != NA 且 contact_pairs 非空，
         则 contact_pairs[0][0]（actor 侧 body part）必须等于 schema.actor_part。
      2. 若 schema.actor_part == NA，跳过断言（允许全 NA schema）。
      3. 若 contact_pairs 为空，跳过断言（无接触 schema 不需要 pair 对齐）。

    同时验证 contact_pairs 第一项属于 body_part 词表，而非 partner 方词条排在前面。

    在 build_phase_sample() 解析完 schema 和 contact_pairs 后调用。
    """
    from .hhi_types import NA as _NA  # 避免循环导入
    if schema.actor_part == _NA or not contact_pairs:
        return

    first_pair = contact_pairs[0]
    if not (isinstance(first_pair, (list, tuple)) and len(first_pair) >= 2):
        return  # 格式不完整时宽松跳过，让上层校验

    actor_bp_in_pair = first_pair[0]  # 约定 contact_pairs = (actor_part, partner_part)

    # 验证词表合法性
    vocabs   = load_schema_vocabs()
    bp_vocab = vocabs["body_part"]
    if actor_bp_in_pair not in bp_vocab:
        raise AssertionError(
            f"contact_pairs[0][0]='{actor_bp_in_pair}' 不在 body_part 词表中。"
        )

    # 验证与 schema.actor_part 一致
    if actor_bp_in_pair != schema.actor_part:
        raise AssertionError(
            f"contact_pairs[0][0]='{actor_bp_in_pair}' != schema.actor_part='{schema.actor_part}' "
            f"(actor_id='{sample_actor_id}'). "
            "contact_pairs 第一项 body part 必须来自当前 actor 的 schema.actor_part。"
        )
