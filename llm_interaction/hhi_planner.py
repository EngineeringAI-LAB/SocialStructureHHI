"""
hhi_planner.py
---------------
LLM Planner：调用 LLM 产生结构化 phase 计划，包含每个 actor 的自然语言动作描述
和 spatial allocator 输出。

核心函数：
  plan_interaction()        -- 调用 LLM，返回 PlannerOutput
  normalize_phase_lengths() -- duration bucket 合法化并回写 PlannerOutput

禁止：直接操作 PyTorch tensor；在此模块做关节级运动生成。
"""
from __future__ import annotations

import collections
import glob as _glob
import json
import logging
import os
import re
import subprocess
import time
from typing import Any, Dict, List, Optional

from hhi.hhi_constants import DURATION_BUCKETS, INTERX_ACTION_NAMES
from hhi.hhi_duration_buckets import bucketize_phase_sequence
from hhi.hhi_types import (
    InteractionMode,
    PhasePlan,
    PlannerOutput,
    SpatialTarget,
)

logger = logging.getLogger(__name__)

# ── Phase bucket table ────────────────────────────────────────────────────────
# interaction_category → tempo → {phase_type → bucket}
# tempo ∈ {"slow", "medium", "fast"}，由 GT real_len 三等分分位数得出：
#   fast   = 下三分位数对应 bucket（最短）
#   medium = 中间三分位数对应 bucket
#   slow   = 上三分位数对应 bucket（最长）
# "_default" 用于 interaction_category 未知时的 fallback。
#
# 用 compute_interaction_phase_bucket_table() 从 .npz 填充并保存为 JSON；
# 推理时 random.choice(["slow","medium","fast"]) 采样一种节奏。
PHASE_BUCKET_TABLE: Dict[str, Dict[str, Dict[str, int]]] = {
    "_default": {
        "approach"             : {"slow": 90,  "medium": 60, "fast": 60},
        "contact"              : {"slow": 120, "medium": 90, "fast": 60},
        "release"              : {"slow": 90,  "medium": 60, "fast": 60},
        "non_contact_interact" : {"slow": 120, "medium": 90, "fast": 60},
    },
}

_TEMPOS = ("slow", "medium", "fast")
_DEFAULT_BUCKET = 60

_PHASE_TYPE_LIST       = ", ".join(f'"{k}"' for k in PHASE_BUCKET_TABLE["_default"])
_ACTION_CATEGORY_LIST  = ", ".join(f'"{n}"' for n in INTERX_ACTION_NAMES)

_SYSTEM_PROMPT = f"""You are a Planner for a two-person motion generation system.
Given a global text description of a two-person interaction, output a structured JSON execution plan.

ARCHITECTURE: In every phase, BOTH P1 and P2 execute motion simultaneously.
  - P1 generates first (Ping), using context from P2's previous phase.
  - P2 generates second (Pong), using P1's just-generated motion as context.
  - Phases represent SEMANTIC TIME WINDOWS — both people are active in every phase.
  - executor_actor indicates who semantically INITIATES or LEADS that phase.

The output format must be exactly:
{{
  "sequence_goal": "...",
  "interaction_category": "Hug",
  "phases": [
    {{
      "phase_id": "phase_000",
      "mode": "INTERACTION_MODE",
      "executor_actor": "P1",
      "phase_type": "approach",
      "tempo": "medium",
      "has_contact": false,
      "spatial_target": {{
        "target_root_distance": 1.2,
        "target_bearing_deg": 0.0
      }},
      "p1_text": "P1 walks toward P2 with arms at sides, closing the distance.",
      "p2_text": "P2 stands still facing P1, watching P1 approach."
    }}
  ]
}}

Rules:
1. Every phase must have BOTH p1_text AND p2_text — one sentence each describing what each person does.
2. p1_text describes P1's motion; p2_text describes P2's motion. Be concrete about body parts and contact.
3. executor_actor must be "P1" or "P2" — indicates the semantic initiator of that phase.
4. phase_type must be one of [{_PHASE_TYPE_LIST}]:
     approach             — one or both people move toward each other (no physical contact)
     contact              — sustained physical contact (hug, hold, handshake, embrace, etc.)
     release              — bodies separate after contact; people move apart or return to neutral
     non_contact_interact — interaction without physical contact (waving, gesturing, mirroring, talking)
5. has_contact is true when P1 and P2 physically touch in this phase.
6. Bilateral contact CAN appear in the same phase:
   p1_text: "P1 raises both arms and wraps them around P2's shoulders."
   p2_text: "P2 wraps their arms around P1's waist, pulling closer."
7. Use fine-grained phases: each major semantic state transition (approach ends, contact begins,
   contact type changes, action ends) should mark a new phase boundary.
8. mode must be "INTERACTION_MODE" or "NAVIGATION_MODE".
9. Keep each text sentence under 25 words and in plain English.
10. target_root_distance is the distance between the two people's root positions (in meters).
    Minimum values by contact type:
      - No contact (waving, gesturing): 0.6 – 1.5 m
      - Light contact (handshake, pat on back): 0.4 – 0.7 m
      - Close contact (hug, embrace): 0.2 – 0.4 m
    Never use 0.0 — two people cannot physically overlap.
11. interaction_category must be exactly one of [{_ACTION_CATEGORY_LIST}].
    Pick the single closest match to the described interaction.
12. tempo must be "slow", "medium", or "fast" — choose based on the emotional quality of that phase:
      slow   — deliberate, tender, ceremonial (e.g. slow embrace, careful approach)
      medium — normal conversational pace
      fast   — urgent, energetic, aggressive (e.g. rushing grab, quick high-five)
    Each phase picks its own tempo independently.
"""

# ── Ollama 服务管理 ────────────────────────────────────────────────────────────

_OLLAMA_BIN  = os.environ.get("OLLAMA_BIN", "/home/wan451/local/ollama/bin/ollama")
_OLLAMA_PORT = int(os.environ.get("OLLAMA_PORT", "11435"))
_ollama_proc: Optional[subprocess.Popen] = None   # 本进程启动的 ollama 子进程


def _ensure_ollama_running() -> None:
    """
    确保本地 Ollama 服务在 _OLLAMA_PORT 上可达。
    若不可达则用 `ollama serve` 启动，等待最多 15 秒。
    """
    global _ollama_proc
    import urllib.request

    health_url = f"http://localhost:{_OLLAMA_PORT}"

    def _reachable() -> bool:
        try:
            urllib.request.urlopen(health_url, timeout=2)
            return True
        except Exception:
            return False

    if _reachable():
        return

    if not os.path.isfile(_OLLAMA_BIN):
        raise RuntimeError(
            f"Ollama 可执行文件不存在：{_OLLAMA_BIN}。"
            "请设置环境变量 OLLAMA_BIN 指向正确路径。"
        )

    logger.info("Ollama 服务未运行，正在启动（port=%d）...", _OLLAMA_PORT)
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"0.0.0.0:{_OLLAMA_PORT}"
    _ollama_proc = subprocess.Popen(
        [_OLLAMA_BIN, "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(15):
        time.sleep(1)
        if _reachable():
            logger.info("Ollama 服务已就绪（port=%d）", _OLLAMA_PORT)
            return

    _ollama_proc.kill()
    raise RuntimeError(
        f"Ollama 服务启动超时（15s），请检查端口 {_OLLAMA_PORT} 是否被占用。"
    )


# ── LLM Backend abstraction ────────────────────────────────────────────────────

def _call_llm(
    prompt          : str,
    model_name      : str = "qwen3.5:35b",
    api_key         : Optional[str] = None,
    max_tokens      : int = 4096,
    temperature     : float = 0.2,
    base_url        : Optional[str] = None,
) -> str:
    """
    调用 LLM API 并返回原始文本。

    本地 Ollama（默认）：使用 Ollama native /api/chat 并设置 think=False，
      完全禁用 Qwen3.5 思考模式。

    远端 OpenAI API：base_url 非 localhost。
    """
    import urllib.request as _urlreq
    import json as _json

    resolved_base_url = base_url or os.environ.get(
        "LLM_BASE_URL", "http://localhost:11435/v1"
    )
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY", "ollama")

    _is_ollama = "localhost" in resolved_base_url or "127.0.0.1" in resolved_base_url

    if _is_ollama:
        if f"localhost:{_OLLAMA_PORT}" in resolved_base_url:
            _ensure_ollama_running()

        port_match = re.search(r":(\d+)", resolved_base_url)
        port = port_match.group(1) if port_match else str(_OLLAMA_PORT)
        ollama_url = f"http://localhost:{port}/api/chat"
        payload = _json.dumps({
            "model"   : model_name,
            "think"   : False,
            "stream"  : False,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "options" : {"temperature": temperature, "num_predict": max_tokens},
        }).encode()
        try:
            req = _urlreq.Request(ollama_url, data=payload,
                                  headers={"Content-Type": "application/json"})
            with _urlreq.urlopen(req, timeout=300) as resp:
                result = _json.loads(resp.read())
            return result.get("message", {}).get("content", "") or ""
        except Exception as e:
            logger.error("Ollama native API call failed: %s", e)
            raise
    else:
        try:
            import openai
            client = openai.OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except ImportError:
            raise ImportError(
                "openai package not installed. Install it with: pip install openai"
            )
        except Exception as e:
            logger.error("LLM API call failed: %s", e)
            raise


def _extract_json(text: str) -> Optional[dict]:
    """从 LLM 输出中提取 JSON（处理 markdown code block 等包装）。"""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    logger.error("无法从 LLM 输出中解析 JSON")
    return None


def _lookup_bucket(interaction_category: str, phase_type: str, tempo: str) -> int:
    """
    查找 bucket：interaction → phase_type → tempo。
    若该 interaction 的该 phase_type 为 None（GT 中不涵盖），
    自动 fallback 到 _default 行，再取 tempo，再取 _DEFAULT_BUCKET。
    """
    interaction_entry = PHASE_BUCKET_TABLE.get(interaction_category) or PHASE_BUCKET_TABLE["_default"]
    tempo_map = interaction_entry.get(phase_type)   # may be None
    if not tempo_map:
        tempo_map = PHASE_BUCKET_TABLE["_default"].get(phase_type) or {}
    return tempo_map.get(tempo, _DEFAULT_BUCKET)


def _dict_to_phase_plan(d: dict, interaction_category: str = "") -> PhasePlan:
    executor = d.get("executor_actor", "P1")
    spatial  = d.get("spatial_target", {})

    raw_mode = d.get("mode", "INTERACTION_MODE")
    try:
        mode = InteractionMode(raw_mode)
    except ValueError:
        mode = InteractionMode.INTERACTION

    phase_type = d.get("phase_type", "approach")
    tempo      = d.get("tempo", "medium") if d.get("tempo") in _TEMPOS else "medium"
    dur_bucket = _lookup_bucket(interaction_category, phase_type, tempo)
    return PhasePlan(
        phase_id             = d.get("phase_id", "phase_000"),
        mode                 = mode,
        executor_actor       = executor,
        phase_type           = phase_type,
        tempo                = tempo,
        interaction_category = interaction_category,
        raw_duration         = dur_bucket,
        duration_bucket      = dur_bucket,
        spatial_target       = SpatialTarget(
            target_root_distance=float(spatial.get("target_root_distance", 1.0)),
            target_bearing_deg  =float(spatial.get("target_bearing_deg", 0.0)),
        ),
        p1_text     = d.get("p1_text", ""),
        p2_text     = d.get("p2_text", ""),
        has_contact = bool(d.get("has_contact", False)),
    )


def _dict_to_planner_output(d: dict) -> PlannerOutput:
    raw_category = d.get("interaction_category", "")
    interaction_category = raw_category if raw_category in INTERX_ACTION_NAMES else ""
    if raw_category and not interaction_category:
        logger.warning("LLM 返回未知 interaction_category=%r，忽略", raw_category)

    phases = [
        _dict_to_phase_plan(p, interaction_category=interaction_category)
        for p in d.get("phases", [])
    ]
    return PlannerOutput(
        sequence_goal = d.get("sequence_goal", ""),
        phases        = phases,
    )


# ── Core planner functions ─────────────────────────────────────────────────────

def plan_interaction(
    global_text  : str,
    model_name   : str = "qwen3.5:35b",
    api_key      : Optional[str] = None,
    temperature  : float = 0.2,
    base_url     : Optional[str] = None,
    max_retries  : int = 3,
) -> PlannerOutput:
    """
    调用 LLM，将全局文本描述转化为结构化 PlannerOutput。

    返回的 plan 已经过 duration bucket normalization。
    若 LLM 返回无效 JSON，自动重试最多 max_retries 次。
    """
    prompt = f"Generate the execution plan for the following two-person interaction:\n\n{global_text}"

    data = None
    for attempt in range(1, max_retries + 1):
        raw_out = _call_llm(prompt, model_name=model_name, api_key=api_key,
                            temperature=temperature, base_url=base_url)
        data = _extract_json(raw_out)
        if data is not None:
            break
        logger.warning("LLM 返回无效 JSON（第 %d/%d 次），重试...", attempt, max_retries)

    if data is None:
        logger.warning("LLM 连续 %d 次返回无效 JSON，使用空计划", max_retries)
        return PlannerOutput(sequence_goal=global_text)

    plan = _dict_to_planner_output(data)
    plan = normalize_phase_lengths(plan)
    return plan


def normalize_phase_lengths(plan: PlannerOutput) -> PlannerOutput:
    """
    对 plan.phases 中每个 phase 的 raw_duration 做 bucket 合法化，
    并将 duration_bucket 原地回写到 phase 对象。

    若触发 phase split，直接回写修正后的 phases 列表。
    """
    phase_dicts = [
        {
            "phase_id"    : p.phase_id,
            "raw_duration": p.raw_duration,
            "has_contact" : p.has_contact,
            "is_splittable": True,
            "t_start"     : 0,
            "_phase_obj"  : p,
        }
        for p in plan.phases
    ]

    bucketized, records = bucketize_phase_sequence(phase_dicts)

    new_phases = []
    for bd in bucketized:
        orig: PhasePlan = bd["_phase_obj"]
        orig.duration_bucket = bd["duration_bucket"]

        if bd.get("is_split_child"):
            import copy
            child = copy.deepcopy(orig)
            child.phase_id        = bd["phase_id"]
            child.raw_duration    = bd["raw_duration"]
            child.duration_bucket = bd["duration_bucket"]
            new_phases.append(child)
        else:
            new_phases.append(orig)

    plan.phases = new_phases
    return plan


# ── GT statistics ──────────────────────────────────────────────────────────────

def load_phase_bucket_table(path: str) -> Dict[str, Dict[str, int]]:
    """从 JSON 文件加载 PHASE_BUCKET_TABLE 并更新模块变量。"""
    global PHASE_BUCKET_TABLE
    with open(path) as f:
        PHASE_BUCKET_TABLE = json.load(f)
    logger.info("PHASE_BUCKET_TABLE 已从 %s 加载（%d 种 interaction）",
                path, len(PHASE_BUCKET_TABLE) - 1)
    return dict(PHASE_BUCKET_TABLE)


def compute_interaction_phase_bucket_table(
    data_dir  : str,
    split     : str = "train",
    save_path : Optional[str] = None,
) -> Dict[str, Dict[str, int]]:
    """
    从已导出的 .npz 训练文件统计 PHASE_BUCKET_TABLE 并原地更新模块变量。

    对每种 (interaction, phase_type) 收集所有 GT real_len，排序后三等分：
      fast   = 下三分位数代表值 → nearest bucket（最短）
      medium = 中三分位数代表值 → nearest bucket
      slow   = 上三分位数代表值 → nearest bucket（最长）
    三等分代表值取各段中位数。

    格式：
        {
          "_default": {
              "approach": {"slow": 90, "medium": 60, "fast": 60}, ...
          },
          "Hug": {
              "approach":     {"slow": 90,  "medium": 60, "fast": 60},
              "contact_hold": {"slow": 120, "medium": 90, "fast": 60},
              ...
          },
          ...
        }

    Args:
        data_dir : 数据集根目录（含 split 子目录）。
        split    : "train" | "val" | "test"。
        save_path: 若指定，将结果写入该 JSON 文件。
    """
    import re as _re
    import numpy as np
    global PHASE_BUCKET_TABLE

    phase_types = list(PHASE_BUCKET_TABLE["_default"].keys())
    split_dir   = os.path.join(data_dir, split)
    npz_files   = _glob.glob(os.path.join(split_dir, "*.npz"))
    if not npz_files:
        logger.warning("在 %s 下未找到 .npz 文件，返回当前表。", split_dir)
        return dict(PHASE_BUCKET_TABLE)

    # 三等分需要至少 3 个样本才有意义；不足时该 phase 标记为 null，
    # 表示该 interaction 在 GT 中不涵盖该 phase，推理时 fallback 到 _default
    MIN_SAMPLES = 3

    def _tertile_buckets(buckets: List[int]) -> Optional[Dict[str, int]]:
        """
        将 phase_len_bucket 列表排序后三等分，各段取中位数。
        样本不足 MIN_SAMPLES 时返回 None（表示该 phase 在此 interaction 无效）。
        """
        if len(buckets) < MIN_SAMPLES:
            return None
        s = sorted(buckets)
        n = len(s)
        t1 = max(1, n // 3)
        t2 = max(t1 + 1, 2 * n // 3)
        return {
            "fast"  : s[t1 // 2],
            "medium": s[(t1 + t2) // 2],
            "slow"  : s[(t2 + n - 1) // 2],
        }

    # (action, phase_type) -> [phase_len_bucket, ...]
    # 直接读训练时实际使用的窗口大小，保证推理 bucket 分布与训练一致
    acc: Dict[tuple, List[int]] = collections.defaultdict(list)

    for npz_path in npz_files:
        try:
            d = np.load(npz_path, allow_pickle=False)
        except Exception as e:
            logger.debug("跳过损坏文件 %s: %s", npz_path, e)
            continue

        if any(k not in d for k in ("sequence_id", "phase_type", "phase_len_bucket")):
            continue

        seq_id     = str(d["sequence_id"].flat[0])
        phase_type = str(d["phase_type"].flat[0])
        bucket     = int(d["phase_len_bucket"].flat[0])

        if phase_type not in phase_types:
            continue

        m = _re.search(r"A(\d+)", seq_id)
        action = (INTERX_ACTION_NAMES[int(m.group(1))]
                  if m and int(m.group(1)) < len(INTERX_ACTION_NAMES) else "")

        acc[(action, phase_type)].append(bucket)

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    # new_table: action → phase_type → {slow/medium/fast → bucket} | null
    # null 表示该 interaction 在 GT 中不涵盖该 phase，推理时 fallback 到 _default
    new_table: Dict[str, Dict[str, Any]] = collections.defaultdict(dict)

    for (action, phase_type), buckets in acc.items():
        result = _tertile_buckets(buckets)
        new_table[action][phase_type] = result   # None = 样本不足，显式标记

    # "" → "_default"
    default_entry = new_table.pop("", {})
    for pt in phase_types:
        # _default 也可能样本不足；此时保留初始硬编码值
        if not default_entry.get(pt):
            default_entry[pt] = PHASE_BUCKET_TABLE["_default"][pt]

    # 对每个 interaction，缺失的 phase_type 显式设为 None（不用 _default 填充）
    # _lookup_bucket 运行时再 fallback，避免用不相关的统计值误导
    for action in new_table:
        for pt in phase_types:
            new_table[action].setdefault(pt, None)

    new_table["_default"] = default_entry
    PHASE_BUCKET_TABLE = dict(new_table)

    n_samples = sum(len(v) for v in acc.values())  # 每个 .npz 贡献一条记录
    logger.info("PHASE_BUCKET_TABLE 已更新（%d 个样本，%d 种 interaction）",
                n_samples, len(new_table) - 1)

    if save_path:
        with open(save_path, "w") as f:
            json.dump(PHASE_BUCKET_TABLE, f, indent=2, ensure_ascii=False)
        logger.info("PHASE_BUCKET_TABLE 已保存到 %s", save_path)

    return dict(PHASE_BUCKET_TABLE)
