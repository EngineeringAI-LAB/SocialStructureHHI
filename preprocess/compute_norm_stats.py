"""
compute_norm_stats.py
---------------------
Compute per-dimension mean and std over the training split of dataset_npz,
and save to data/motion_norm_stats.npz.

Run after rebuilding dataset_npz:
  python preprocess/compute_norm_stats.py
"""
from __future__ import annotations

import glob
import logging
import os
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_REPO      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR  = os.path.join(_REPO, "data", "dataset_npz", "train")
OUT_PATH   = os.path.join(_REPO, "data", "motion_norm_stats.npz")
MOTION_KEY = "target_motion_local"


def main(train_dirs, out_path=OUT_PATH):
    files = []
    for d in train_dirs:
        files.extend(glob.glob(os.path.join(d, "**", "*.npz"), recursive=True))
    files = sorted(files)
    logger.info("Found %d train npz files across %d dirs", len(files), len(train_dirs))
    if not files:
        sys.exit("No files found – run export scripts first.")

    # Online Welford algorithm for numerically stable mean/var
    count = np.zeros(201, dtype=np.float64)
    mean  = np.zeros(201, dtype=np.float64)
    M2    = np.zeros(201, dtype=np.float64)

    skipped = 0
    for i, path in enumerate(files):
        d = np.load(path, allow_pickle=True)
        if MOTION_KEY not in d.files:
            skipped += 1
            continue
        motion   = d[MOTION_KEY].astype(np.float64)   # [W, 201]
        seq_mask = d.get("seq_mask", np.ones(motion.shape[0], dtype=bool))
        valid    = motion[seq_mask.astype(bool)]       # only valid frames
        if valid.shape[0] == 0:
            skipped += 1
            continue

        for frame in valid:
            count  += 1
            delta   = frame - mean
            mean   += delta / count
            M2     += delta * (frame - mean)

        if (i + 1) % 2000 == 0:
            logger.info("  processed %d / %d files", i + 1, len(files))

    if skipped:
        logger.warning("Skipped %d files (missing key or empty)", skipped)

    total_frames = int(count[0])
    logger.info("Total valid frames: %d", total_frames)

    std = np.sqrt(M2 / np.maximum(count - 1, 1)).astype(np.float32)
    std = np.clip(std, 1e-3, None)
    mean = mean.astype(np.float32)

    np.savez_compressed(out_path, mean=mean, std=std)
    logger.info("Saved norm stats → %s  (dim=201, frames=%d)", out_path, total_frames)

    # Quick sanity check
    logger.info("mean range: [%.4f, %.4f]", mean.min(), mean.max())
    logger.info("std  range: [%.4f, %.4f]", std.min(),  std.max())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dirs", nargs="+", default=[TRAIN_DIR],
                        help="One or more train split directories")
    parser.add_argument("--out_path",   default=OUT_PATH)
    args = parser.parse_args()
    main(train_dirs=args.train_dirs, out_path=args.out_path)
