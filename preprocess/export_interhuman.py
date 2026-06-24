"""
export_interhuman.py
--------------------
Export the InterHuman dataset to the ego-centric per-phase .npz format used to
train the executor. Ported from the original `export_interhuman_*.sh` SLURM heredoc.

Same pipeline as export_interx.py but using the InterHuman loaders. Run the LLM
annotation stage first (preprocess/annotate_interhuman.py) so the cache exists.

  python preprocess/export_interhuman.py --split train \
      --out_dir data/dataset_npz --cache_dir data/llm_annot_cache \
      --shard_id $SLURM_ARRAY_TASK_ID --num_shards $SLURM_ARRAY_TASK_COUNT

InterHuman dataset paths are configured in hhi/interhuman_dataset_builder.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hhi.hhi_dataset_builder import (
    _save_sample, annotate_phases_with_llm, build_phase_sample,
    replay_and_label_modes, segment_interaction_phases,
)
from hhi.interhuman_dataset_builder import (
    action_name_interhuman, compute_world_joints_interhuman,
    load_interhuman_sequence, load_interhuman_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

FOOT_INDICES = [7, 8, 10, 11]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--out_dir", default="data/dataset_npz")
    ap.add_argument("--cache_dir", default="data/llm_annot_cache")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    seq_ids = load_interhuman_split(args.split)
    seq_ids = seq_ids[args.shard_id::args.num_shards]
    out_dir = os.path.join(args.out_dir, args.split, f"ih_task_{args.shard_id:02d}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info("InterHuman %s shard %d/%d: %d sequences -> %s",
                args.split, args.shard_id, args.num_shards, len(seq_ids), out_dir)

    stats = {"n_sequences": len(seq_ids), "n_phases": 0, "n_samples": 0,
             "fallback_seqs": 0, "contact_sample_count": 0}

    for seq_id_raw in seq_ids:
        try:
            seq = load_interhuman_sequence(seq_id_raw)
            seq = compute_world_joints_interhuman(seq, device="cuda")
        except Exception as e:
            logger.error("skip %s: %s", seq_id_raw, e)
            continue

        mode_labels, _ = replay_and_label_modes(seq, FOOT_INDICES)
        phases = segment_interaction_phases(seq, mode_labels)
        if not phases:
            stats["fallback_seqs"] += 1
            continue
        try:
            phases = annotate_phases_with_llm(phases, seq, llm_cache_path=args.cache_dir)
        except Exception as e:
            logger.error("annotate failed %s: %s", seq_id_raw, e)

        interaction_category = action_name_interhuman(seq["seq_id"])
        prev_phase_meta = None
        for phase_meta in phases:
            stats["n_phases"] += 1
            for actor_id in ["P1", "P2"]:
                phase_text = phase_meta.get("p1_text" if actor_id == "P1" else "p2_text", "")
                try:
                    sample = build_phase_sample(
                        actor_id=actor_id, phase_meta=phase_meta,
                        prev_phase_meta=prev_phase_meta, sequence=seq,
                        foot_joint_indices=FOOT_INDICES, phase_text=phase_text,
                        phase_type=phase_meta.get("phase_type", "approach"),
                        interaction_category=interaction_category,
                    )
                    if sample.contact_frame_mask is not None and sample.contact_frame_mask.any():
                        stats["contact_sample_count"] += 1
                    _save_sample(sample, os.path.join(out_dir, f"{sample.phase_id}_{actor_id}.npz"))
                    stats["n_samples"] += 1
                except Exception as e:
                    logger.error("build_phase_sample %s phase=%s actor=%s: %s",
                                 seq_id_raw, phase_meta.get("phase_id"), actor_id, e)
            prev_phase_meta = phase_meta

    with open(os.path.join(out_dir, "task_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Done: %d samples from %d phases", stats["n_samples"], stats["n_phases"])


if __name__ == "__main__":
    main()
