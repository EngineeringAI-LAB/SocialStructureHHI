"""
annotate_worker.py
------------------
单个 SLURM job 的标注 worker：读取 seq_id 列表，对每条序列做
  1. load_sequence_from_h5
  2. compute_world_joints（SMPL-X FK）
  3. replay_and_label_modes + segment_interaction_phases
  4. annotate_phases_with_llm（多轮 KV-cache session）

结果缓存到 --cache_dir，已缓存的序列自动跳过。
"""
import argparse
import json
import logging
import os
import sys
import time
import traceback

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hhi.hhi_dataset_builder import (
    annotate_phases_with_llm,
    compute_world_joints,
    load_sequence_from_h5,
    replay_and_label_modes,
    segment_interaction_phases,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_list",   required=True,  help="文件，每行一个 seq_id")
    parser.add_argument("--cache_dir",  required=True,  help="标注结果缓存目录")
    parser.add_argument("--llm_model",      default="qwen3.5:35b")
    parser.add_argument("--llm_port",       type=int, default=11500)
    parser.add_argument("--job_id",         type=int, default=0, help="SLURM_ARRAY_TASK_ID")
    parser.add_argument("--no_motion_facts", action="store_true",
                        help="ablation: skip MOTION FACTS in LLM prompt")
    args = parser.parse_args()

    llm_base_url = f"http://127.0.0.1:{args.llm_port}/v1"

    with open(args.seq_list) as f:
        seq_ids = [l.strip() for l in f if l.strip()]

    total   = len(seq_ids)
    done    = 0
    skipped = 0
    failed  = 0

    print(f"[job {args.job_id}] {total} sequences  cache={args.cache_dir}  model={args.llm_model}")
    sys.stdout.flush()

    # Load body model once and reuse across all sequences
    import torch
    from human_body_prior.body_model.body_model import BodyModel
    from hhi.hhi_dataset_builder import _find_smplx_model_path
    import os as _os
    _bm_path = _os.environ.get("SMPLX_MODEL_PATH", _find_smplx_model_path())
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    body_model = BodyModel(bm_fname=_bm_path, num_betas=10).to(_device)
    body_model.eval()
    print(f"[job {args.job_id}] Body model loaded on {_device}")
    sys.stdout.flush()

    bar = tqdm(
        seq_ids,
        desc=f"job{args.job_id:02d}",
        unit="seq",
        dynamic_ncols=True,
        file=sys.stdout,
    )

    _LAZY = {
        "P1 moves.", "P2 moves.",
        "P1 makes physical contact with P2.",
        "P2 makes physical contact with P1.",
        "Person 1 moves.", "Person 2 moves.",
        "Person 1 makes physical contact with Person 2.",
        "Person 2 makes physical contact with Person 1.",
    }

    def _cache_is_complete(cache_path: str) -> bool:
        """Return True only if cache file exists and contains no lazy/fallback entries."""
        if not os.path.exists(cache_path):
            return False
        try:
            with open(cache_path) as f:
                data = json.load(f)
            phases = data.get("local_phases", data)
            if not phases:
                return False
            for entry in phases.values():
                p1 = entry.get("p1_action", "").strip()
                p2 = entry.get("p2_action", "").strip()
                if (not p1 or p1 in _LAZY) and (not p2 or p2 in _LAZY):
                    return False   # lazy entry found — needs re-annotation
            return True
        except Exception:
            return False

    for seq_id in bar:
        # 已完整标注（无 lazy 条目）则跳过
        cache_file = os.path.join(args.cache_dir, f"{seq_id}.json")
        if _cache_is_complete(cache_file):
            skipped += 1
            bar.set_postfix(done=done, skip=skipped, fail=failed)
            continue

        try:
            seq = load_sequence_from_h5(seq_id)
            seq = compute_world_joints(seq, body_model=body_model, device=_device)
            mode_labels, _ = replay_and_label_modes(seq)
            phases = segment_interaction_phases(seq, mode_labels)

            if not phases:
                logger.warning("%s: no phases found, skipping", seq_id)
                failed += 1
                bar.set_postfix(done=done, skip=skipped, fail=failed)
                continue

            phases = annotate_phases_with_llm(
                phases, seq,
                llm_model=args.llm_model,
                llm_base_url=llm_base_url,
                llm_cache_path=args.cache_dir,
                no_motion_facts=args.no_motion_facts,
            )
            done += 1

        except Exception:
            logger.error("%s: annotation failed\n%s", seq_id, traceback.format_exc())
            failed += 1

        bar.set_postfix(done=done, skip=skipped, fail=failed)

    bar.close()
    print(f"[job {args.job_id}] DONE: {done} annotated, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
