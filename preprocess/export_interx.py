"""
export_interx.py
----------------
Export the Inter-X dataset to the ego-centric per-phase .npz format used to train
the executor. Ported from the original `export_dataset_*.sh` SLURM heredoc.

For each sequence:
  load_sequence_from_h5 -> compute_world_joints (SMPL-X FK) -> replay_and_label_modes
  -> segment_interaction_phases -> annotate_phases_with_llm (reads LLM cache, no live
  call) -> build_phase_sample (per actor) -> save {phase_id}_{P1,P2}.npz

Run the LLM annotation stage first (preprocess/annotate_interx.py) so the cache exists.

Shardable for SLURM array jobs:
  python preprocess/export_interx.py --split train \
      --out_dir data/dataset_npz --cache_dir data/llm_annot_cache \
      --shard_id $SLURM_ARRAY_TASK_ID --num_shards $SLURM_ARRAY_TASK_COUNT

Dataset paths are configured in hhi/hhi_constants.py (INTERX_*).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hhi.hhi_constants import INTERX_H5_PATH, INTERX_SPLIT_ROOT
from hhi.hhi_dataset_builder import (
    _action_name_from_seq_id, _save_sample, annotate_phases_with_llm,
    build_phase_sample, compute_world_joints, load_sequence_from_h5,
    replay_and_label_modes, segment_interaction_phases,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

FOOT_INDICES = [7, 8, 10, 11]   # SMPL-X L/R ankle, L/R toe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--out_dir", default="data/dataset_npz")
    ap.add_argument("--cache_dir", default="data/llm_annot_cache")
    ap.add_argument("--skip_file", default="", help="optional file of seq_ids to skip")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    split_file = os.path.join(INTERX_SPLIT_ROOT, args.split + ".txt")
    with open(split_file) as f:
        seq_ids = sorted(l.strip() for l in f if l.strip())

    if args.skip_file and os.path.exists(args.skip_file):
        skip = {l.strip() for l in open(args.skip_file) if l.strip()}
        seq_ids = [s for s in seq_ids if s not in skip]

    seq_ids = seq_ids[args.shard_id::args.num_shards]
    out_dir = os.path.join(args.out_dir, args.split, f"task_{args.shard_id:02d}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Inter-X %s shard %d/%d: %d sequences -> %s",
                args.split, args.shard_id, args.num_shards, len(seq_ids), out_dir)

    stats = {"n_sequences": len(seq_ids), "n_phases": 0, "n_samples": 0,
             "fallback_seqs": 0, "contact_sample_count": 0}

    for seq_id in seq_ids:
        try:
            seq = load_sequence_from_h5(seq_id, INTERX_H5_PATH)
            seq = compute_world_joints(seq, body_model=None, device="cuda")
        except Exception as e:
            logger.error("skip seq %s: %s", seq_id, e)
            continue

        mode_labels, _ = replay_and_label_modes(seq, FOOT_INDICES)
        phases = segment_interaction_phases(seq, mode_labels, annotation_path=None, keyword_map_path=None)
        if not phases:
            stats["fallback_seqs"] += 1
            continue
        try:
            phases = annotate_phases_with_llm(phases, seq, llm_cache_path=args.cache_dir)
        except Exception as e:
            logger.error("annotate_phases_with_llm failed for %s: %s", seq_id, e)

        interaction_category = _action_name_from_seq_id(seq_id)
        prev_phase_meta = None
        seq_phase0_meta = None
        for phase_meta in phases:
            stats["n_phases"] += 1
            for actor_id in ["P1", "P2"]:
                phase_text = phase_meta.get("p1_text" if actor_id == "P1" else "p2_text", "")
                try:
                    sample = build_phase_sample(
                        actor_id=actor_id, phase_meta=phase_meta,
                        prev_phase_meta=prev_phase_meta, phase0_meta=seq_phase0_meta,
                        sequence=seq, foot_joint_indices=FOOT_INDICES,
                        phase_text=phase_text, phase_type=phase_meta.get("phase_type", "approach"),
                        interaction_category=interaction_category,
                    )
                    if sample.contact_frame_mask is not None and sample.contact_frame_mask.any():
                        stats["contact_sample_count"] += 1
                    _save_sample(sample, os.path.join(out_dir, f"{sample.phase_id}_{actor_id}.npz"))
                    stats["n_samples"] += 1
                except Exception as e:
                    logger.error("build_phase_sample failed seq=%s phase=%s actor=%s: %s",
                                 seq_id, phase_meta.get("phase_id"), actor_id, e)
            if seq_phase0_meta is None:
                seq_phase0_meta = phase_meta
            prev_phase_meta = phase_meta

    with open(os.path.join(out_dir, "task_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Done: %d samples from %d phases", stats["n_samples"], stats["n_phases"])


if __name__ == "__main__":
    main()
