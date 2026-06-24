"""
annotate_interhuman_worker.py
------------------------------
LLM annotation worker for InterHuman sequences.
Same logic as annotate_worker.py but loads via interhuman_dataset_builder.

seq_list file contains raw integer IDs (e.g. "1042"); cache files are written
as "IH_1042.json" to share the same cache dir without collision with Inter-X.
"""
import argparse
import json
import logging
import os
import sys
import traceback

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hhi.interhuman_dataset_builder import (
    compute_world_joints_interhuman,
    load_interhuman_sequence,
    SEQ_ID_PREFIX,
)
from hhi.hhi_dataset_builder import (
    annotate_phases_with_llm,
    replay_and_label_modes,
    segment_interaction_phases,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FOOT_INDICES = [7, 8, 10, 11]   # L_Ankle, R_Ankle, L_Foot, R_Foot in SMPL body joints

_LAZY = {
    "P1 moves.", "P2 moves.",
    "P1 makes physical contact with P2.",
    "P2 makes physical contact with P1.",
    "Person 1 moves.", "Person 2 moves.",
    "Person 1 makes physical contact with Person 2.",
    "Person 2 makes physical contact with Person 1.",
}


def _cache_is_complete(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
        phases = data.get("local_phases", data)
        if not phases:
            return False
        for entry in phases.values():
            p1 = entry.get("p1_action", "").strip()
            p2 = entry.get("p2_action", "").strip()
            if (not p1 or p1 in _LAZY) and (not p2 or p2 in _LAZY):
                return False
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_list",  required=True, help="文件，每行一个 raw seq_id (e.g. '1042')")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--llm_model",      default="qwen3.5:35b")
    parser.add_argument("--llm_port",       type=int, default=11500)
    parser.add_argument("--job_id",         type=int, default=0)
    parser.add_argument("--no_motion_facts", action="store_true",
                        help="ablation: skip MOTION FACTS in LLM prompt")
    args = parser.parse_args()

    llm_base_url = f"http://127.0.0.1:{args.llm_port}/v1"

    with open(args.seq_list) as f:
        seq_ids_raw = [l.strip() for l in f if l.strip()]

    total = len(seq_ids_raw)
    done = skipped = failed = 0
    print(f"[job {args.job_id}] {total} InterHuman sequences  cache={args.cache_dir}")
    sys.stdout.flush()

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

    bar = tqdm(seq_ids_raw, desc=f"IH-job{args.job_id:02d}", unit="seq",
               dynamic_ncols=True, file=sys.stdout)

    for seq_id_raw in bar:
        seq_id     = SEQ_ID_PREFIX + seq_id_raw
        cache_file = os.path.join(args.cache_dir, f"{seq_id}.json")

        if _cache_is_complete(cache_file):
            skipped += 1
            bar.set_postfix(done=done, skip=skipped, fail=failed)
            continue

        try:
            seq = load_interhuman_sequence(seq_id_raw)
            seq = compute_world_joints_interhuman(seq, body_model=body_model, device=_device)
            mode_labels, _ = replay_and_label_modes(seq, FOOT_INDICES)
            phases = segment_interaction_phases(seq, mode_labels)

            if not phases:
                logger.warning("%s: no phases found", seq_id)
                failed += 1
                bar.set_postfix(done=done, skip=skipped, fail=failed)
                continue

            annotate_phases_with_llm(
                phases, seq,
                llm_model=args.llm_model,
                llm_base_url=llm_base_url,
                llm_cache_path=args.cache_dir,
                no_motion_facts=args.no_motion_facts,
            )
            done += 1

        except Exception:
            logger.error("%s: failed\n%s", seq_id, traceback.format_exc())
            failed += 1

        bar.set_postfix(done=done, skip=skipped, fail=failed)

    bar.close()
    print(f"[job {args.job_id}] DONE: {done} annotated, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
