"""
hhi_schema_prompt.py
---------------------
离线 LLM 批量转换工具：将 Inter-X 原始文本描述批量转换为结构化 Schema，
输出写入 schema_annotation_cache.json。

用法（在数据集构建前运行一次）：
  python -m llm_interaction.hhi_schema_prompt \
    --text_dir /path/to/inter-x/texts \
    --output   /path/to/schema_annotation_cache.json \
    --model    gpt-4o
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_BATCH_PROMPT_TEMPLATE = """\
下面是 {n} 条双人动作描述，请为每条输出结构化 Schema JSON。

词表（所有字段只能从中取值，不在词表内的写 "[N/A]"）：
  action: reach, hug, handshake, pat, push, pull, point, wave, bow, lean,
          step_toward, step_back, turn, hold, release, kick, punch, dodge,
          embrace, lift, lower, tap, stroke, grab, give, receive, dance, fight, wait, [N/A]
  body_part: RightHand, LeftHand, RightArm, LeftArm, RightForearm, LeftForearm,
             RightShoulder, LeftShoulder, Head, Chest, Spine, Hips,
             RightThigh, LeftThigh, RightLeg, LeftLeg, RightFoot, LeftFoot,
             UpperBody, LowerBody, WholeBody, RightSide, LeftSide, Back, Front, [N/A]
  spatial: face_to_face, side_by_side, behind, in_front, above, below,
           shoulder_height, chest_height, waist_height, head_height,
           close_range, mid_range, far_range, left_of, right_of, [N/A]
  contact: no_contact, light_touch, firm_grasp, full_body_contact, hand_contact,
           shoulder_contact, back_contact, chest_contact, head_contact, arm_contact, foot_contact, [N/A]
  orientation: face_to_face, side_by_side_parallel, side_by_side_opposite,
               behind_target, in_front_of_target, perpendicular, any, [N/A]
  priority: contact, spatial, gesture, safety, [N/A]

actor 是句子主语（发起动作的人）。

动作描述列表：
{descriptions}

请输出一个 JSON 数组，每个元素对应一条描述（保持顺序）：
[
  {{
    "text": "原始描述",
    "action": "...",
    "actor_part": "...",
    "target_part": "...",
    "spatial": "...",
    "contact": "...",
    "orientation": "...",
    "priority": "..."
  }},
  ...
]
"""


def collect_unique_texts(text_dir: str) -> List[str]:
    """扫描 Inter-X texts/ 目录，收集所有去重动作描述。"""
    texts = set()
    for fpath in glob.glob(os.path.join(text_dir, "**", "*.txt"), recursive=True):
        try:
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(None, 2)
                    if len(parts) >= 3:
                        texts.add(parts[2].strip())
                    elif len(parts) >= 1 and len(parts[0]) > 3:
                        texts.add(parts[0].strip())
        except Exception:
            pass
    return sorted(texts)


def _call_llm_batch(
    descriptions : List[str],
    model_name   : str = "gpt-4o",
    api_key      : Optional[str] = None,
) -> Optional[List[dict]]:
    """调用 LLM 批量转换，返回结果列表；失败返回 None。"""
    try:
        import openai
        client  = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        prompt  = _BATCH_PROMPT_TEMPLATE.format(
            n=len(descriptions),
            descriptions="\n".join(f"{i+1}. {t}" for i, t in enumerate(descriptions)),
        )
        resp    = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        # 去掉 markdown
        import re
        raw = re.sub(r"```(?:json)?\s*", "", raw)
        raw = re.sub(r"```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        logger.error("LLM batch call failed: %s", e)
        return None


def build_schema_cache(
    text_dir    : str,
    output_path : str,
    model_name  : str = "gpt-4o",
    api_key     : Optional[str] = None,
    batch_size  : int = 20,
    delay_sec   : float = 1.0,
) -> None:
    """
    主函数：批量转换所有唯一文本描述 → schema，写入 cache JSON。
    """
    texts   = collect_unique_texts(text_dir)
    logger.info("收集到 %d 个唯一动作描述", len(texts))

    # 加载已有 cache（支持断点续传）
    cache: Dict[str, dict] = {}
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            cache = json.load(f)
        logger.info("加载已有 cache：%d 条", len(cache))

    # 过滤已有
    todo = [t for t in texts if t not in cache]
    logger.info("待转换：%d 条", len(todo))

    # 分批处理
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        results = _call_llm_batch(batch, model_name, api_key)
        if results is None:
            logger.warning("batch %d~%d 失败，跳过", start, start + len(batch))
            continue

        for orig_text, item in zip(batch, results):
            if isinstance(item, dict):
                cache[orig_text] = {
                    k: v for k, v in item.items() if k != "text"
                }

        # 每批写一次，防止丢失
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

        logger.info("进度 %d/%d，cache 已保存", min(start + batch_size, len(todo)), len(todo))
        time.sleep(delay_sec)

    logger.info("转换完成，共 %d 条", len(cache))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_dir", required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--model",    default="gpt-4o")
    parser.add_argument("--batch_size", type=int, default=20)
    args = parser.parse_args()
    build_schema_cache(
        text_dir   = args.text_dir,
        output_path= args.output,
        model_name = args.model,
        batch_size = args.batch_size,
    )
