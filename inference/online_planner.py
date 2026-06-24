"""
online_planner.py
------------------
Receding-horizon 在线规划器：每个 phase 执行完后，LLM 根据当前进度规划下一 phase。

与 hhi_planner.py 的区别：
  - hhi_planner.py：一次性规划全部 phase（batch planning）
  - online_planner.py：每步只规划下一个 phase，看到实际执行结果后再决定后续行动

核心类：
  OnlinePlanner    -- 持有 LLM 连接，提供 plan_next() 接口
  CompletedPhase   -- 已执行 phase 的简要记录（供 LLM 参考历史）
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 合法的 bucket 值
_VALID_BUCKETS = [30, 60, 90, 120, 150, 180, 240, 300]
_VALID_PHASE_TYPES = ["approach", "reach", "contact_hold", "release", "step_back", "navigate"]

# ── 系统提示（在线规划专用）─────────────────────────────────────────────────────

_ONLINE_SYSTEM_PROMPT = """You are an online motion planner for a two-person interaction generation system.

You receive the overall interaction goal and a summary of what has been executed so far.
Your job: plan the NEXT single phase only.

Both P1 and P2 move in every phase. P1 generates first (Ping), P2 reacts to P1 (Pong).

Phase types:
  approach       - one or both actors move closer to each other
  reach          - one actor extends a body part toward the other (pre-contact)
  contact_hold   - physical contact is maintained (hug, handshake, pat, etc.)
  release        - contact is being released, bodies separating
  step_back      - one or both actors step away after contact
  navigate       - long-distance travel (rare, use only if >2m apart)

Bucket = number of frames at 30fps. Choose from: 30, 60, 90, 120, 150, 180, 240, 300.
  - Short actions (reach, release): 30–60
  - Medium actions (approach, step_back): 60–90
  - Long actions (contact_hold, slow approach): 90–180

Rules:
  - p1_text and p2_text must be concrete body-motion descriptions (not abstract feelings)
  - Include body parts, direction, speed where relevant
  - Set is_done=true ONLY when the full interaction goal has been achieved
  - Do NOT repeat the same phase_type more than 3 times in a row

Output ONLY valid JSON, no markdown:
{
  "phase_type": "contact_hold",
  "bucket": 120,
  "p1_text": "Person 1 wraps both arms around person 2's torso and holds firmly",
  "p2_text": "Person 2 returns the embrace, placing both arms around person 1's back",
  "rationale": "Main hug phase — the core of the interaction",
  "is_done": false
}"""


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class CompletedPhase:
    """已执行 phase 的简要记录（文本层面，供 LLM 参考）。"""
    phase_idx  : int
    phase_type : str
    bucket     : int
    p1_text    : str
    p2_text    : str
    elapsed_frames: int   # 该 phase 结束时的全局帧数


@dataclass
class NextPhase:
    """LLM 规划的下一个 phase。"""
    phase_type : str
    bucket     : int
    p1_text    : str
    p2_text    : str
    rationale  : str = ""
    is_done    : bool = False


# ── OnlinePlanner ─────────────────────────────────────────────────────────────

class OnlinePlanner:
    """
    Receding-horizon 在线规划器。

    每次调用 plan_next() 时：
      1. 构建包含全局目标 + 已完成历史 + 当前状态的 prompt
      2. 调用 LLM（ollama / OpenAI API）
      3. 解析并验证输出
      4. 返回 NextPhase（or None if is_done）

    参数：
      model_name    : LLM 模型名称（ollama: "qwen3.5:35b"，OpenAI: "gpt-4o"）
      base_url      : API base URL（None 时读 LLM_BASE_URL 环境变量）
      api_key       : API key（None 时读 OPENAI_API_KEY）
      temperature   : LLM 采样温度
      max_phases    : 安全上限，超过后强制终止
    """

    def __init__(
        self,
        model_name  : str   = "qwen3.5:35b",
        base_url    : Optional[str] = None,
        api_key     : Optional[str] = None,
        temperature : float = 0.7,
        max_phases  : int   = 12,
    ):
        self.model_name  = model_name
        self.base_url    = base_url or os.environ.get("LLM_BASE_URL", "http://localhost:11435/v1")
        self.api_key     = api_key  or os.environ.get("OPENAI_API_KEY", "ollama")
        self.temperature = temperature
        self.max_phases  = max_phases

    def plan_next(
        self,
        global_text     : str,
        completed       : List[CompletedPhase],
        current_dist_m  : float = 1.0,
        elapsed_frames  : int   = 0,
    ) -> Optional[NextPhase]:
        """
        根据全局目标和已完成历史，规划下一个 phase。

        返回 None 表示交互已完成（LLM 设置了 is_done=True）或达到 max_phases 上限。
        """
        if len(completed) >= self.max_phases:
            logger.info("Reached max_phases=%d, stopping.", self.max_phases)
            return None

        prompt = self._build_prompt(global_text, completed, current_dist_m, elapsed_frames)
        logger.debug("Online planner prompt:\n%s", prompt)

        try:
            raw = self._llm_call(prompt)
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return None

        phase = self._parse_response(raw)
        if phase is None:
            logger.error("Failed to parse LLM response:\n%s", raw)
            return None

        if phase.is_done:
            logger.info("LLM signaled is_done after %d phases.", len(completed))
            return None

        logger.info(
            "Planned next phase: type=%s bucket=%d  P1: %s",
            phase.phase_type, phase.bucket, phase.p1_text[:60],
        )
        return phase

    # ── Prompt construction ───────────────────────────────────────────────────

    def _build_prompt(
        self,
        global_text    : str,
        completed      : List[CompletedPhase],
        current_dist_m : float,
        elapsed_frames : int,
    ) -> str:
        lines = []
        lines.append(f'Overall interaction goal: "{global_text}"')
        lines.append(f"Current state: distance={current_dist_m:.2f}m, elapsed={elapsed_frames} frames ({elapsed_frames/30:.1f}s)")
        lines.append("")

        if not completed:
            lines.append("No phases executed yet. This will be phase 1.")
        else:
            lines.append(f"Completed phases ({len(completed)} total):")
            for cp in completed:
                lines.append(
                    f"  Phase {cp.phase_idx + 1} [{cp.phase_type}, {cp.bucket} frames]: "
                    f"P1: {cp.p1_text[:80]} | P2: {cp.p2_text[:80]}"
                )

        lines.append("")
        lines.append("Plan the NEXT phase. Output JSON only.")
        return "\n".join(lines)

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _llm_call(self, prompt: str) -> str:
        """调用 LLM，返回原始文本响应。支持 ollama 和 OpenAI 兼容 API。"""
        is_local = "localhost" in self.base_url or "127.0.0.1" in self.base_url

        if is_local:
            return self._call_ollama(prompt)
        else:
            return self._call_openai(prompt)

    def _call_ollama(self, prompt: str) -> str:
        import json as _json
        import urllib.request as _urlreq

        port_match = re.search(r":(\d+)", self.base_url)
        port = port_match.group(1) if port_match else "11435"
        url = f"http://localhost:{port}/api/chat"

        payload = _json.dumps({
            "model"   : self.model_name,
            "think"   : False,
            "stream"  : False,
            "messages": [
                {"role": "system", "content": _ONLINE_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "options": {"temperature": self.temperature, "num_predict": 512},
        }).encode()

        req = _urlreq.Request(url, data=payload,
                              headers={"Content-Type": "application/json"})
        with _urlreq.urlopen(req, timeout=120) as resp:
            result = _json.loads(resp.read())
        return result.get("message", {}).get("content", "") or ""

    def _call_openai(self, prompt: str) -> str:
        import openai
        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": _ONLINE_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=512,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse_response(self, text: str) -> Optional[NextPhase]:
        # 去除 markdown 代码块包装
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```", "", text).strip()

        # 尝试提取 JSON
        d = None
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    d = json.loads(m.group())
                except Exception:
                    pass

        if d is None:
            return None

        # 校验并修正字段
        phase_type = d.get("phase_type", "approach")
        if phase_type not in _VALID_PHASE_TYPES:
            logger.warning("Unknown phase_type %r, defaulting to 'approach'", phase_type)
            phase_type = "approach"

        bucket = int(d.get("bucket", 90))
        if bucket not in _VALID_BUCKETS:
            # 取最近的合法 bucket
            bucket = min(_VALID_BUCKETS, key=lambda b: abs(b - bucket))
            logger.warning("Invalid bucket, snapped to %d", bucket)

        return NextPhase(
            phase_type = phase_type,
            bucket     = bucket,
            p1_text    = str(d.get("p1_text", "")),
            p2_text    = str(d.get("p2_text", "")),
            rationale  = str(d.get("rationale", "")),
            is_done    = bool(d.get("is_done", False)),
        )
