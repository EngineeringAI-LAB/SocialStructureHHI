"""
run_inference.py
----------------
Full text -> two-person-motion online pipeline for run_030.

Pipeline:
  1. Pick a high-level `global_text`. To stay in the training text distribution it
     must come from the demo list (inference/assets/demo_global_texts.json), which
     is sampled from the two datasets the model was trained on (Inter-X / InterHuman).
     A custom prompt can be forced with --allow_custom_global (out-of-distribution).
  2. OnlinePlanner (LLM) decomposes the goal into phases, one at a time, each with
     concrete P1 / P2 body-motion descriptions and a duration bucket.
  3. Each phase text is encoded online with Qwen (+ CLIP) exactly as in training.
  4. The run_030 executor generates the phase Ping-Pong style:
       P1 (Ping): self-history + partner = P2's previous phase
       P2 (Pong): self-history + partner = P1's just-generated phase
     with ego<->world tracking carried across phases.
  5. Output: per-person world motion .npz + an optional standalone HTML viewer.

Phase 0 has no ground-truth start, so the initial pose / facing configuration is
seeded from inference/assets/init_pose.npz (two people standing ~1.3 m apart).

Example:
  python inference/run_inference.py \
      --ckpt checkpoints/run_030/best.pt \
      --demo_id 4 \
      --llm_model qwen3.5:35b --llm_base_url http://localhost:11435/v1 \
      --out_dir outputs/demo_hug --render
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List, Optional

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from inference import infer_core as C
from inference.infer_core import hy_root
from inference.online_planner import CompletedPhase, OnlinePlanner
from inference.text_encoders import CLIPTextEncoder, QwenTextEncoder
from hhi.hhi_partner_model import HHIPartnerModel

# HunyuanMotion repo on sys.path (backbone + body model).
_HY = hy_root()
if _HY not in sys.path:
    sys.path.insert(0, _HY)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ASSETS   = os.path.join(_REPO, "inference", "assets")
OVERLAP_N = C.OVERLAP_N


# ── Demo text list (the only allowed global inputs by default) ────────────────

def load_demo_list() -> List[dict]:
    with open(os.path.join(ASSETS, "demo_global_texts.json")) as f:
        return json.load(f)


def resolve_global_text(args, demo: List[dict]) -> str:
    if args.global_text:
        if not args.allow_custom_global:
            raise SystemExit(
                "Custom --global_text is out-of-distribution for run_030. "
                "Pick one from the demo list with --demo_id, or pass "
                "--allow_custom_global to override (quality may degrade)."
            )
        logger.warning("Using CUSTOM global_text (out-of-distribution): %s", args.global_text)
        return args.global_text
    if args.demo_id is None:
        logger.info("Available demo global texts (use --demo_id N):")
        for i, e in enumerate(demo):
            logger.info("  [%2d] (%s) %s", i, e["dataset"], e["global_text"][:100])
        raise SystemExit("Specify --demo_id N (or --list to print and exit).")
    e = demo[args.demo_id]
    logger.info("global_text [demo %d, %s/%s]: %s", args.demo_id, e["dataset"], e["seq_id"], e["global_text"])
    return e["global_text"]


# ── Initial configuration (Phase 0 has no GT) ─────────────────────────────────

def load_init() -> dict:
    d = np.load(os.path.join(ASSETS, "init_pose.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


# ── One full episode ──────────────────────────────────────────────────────────

def run_episode(
    model      : HHIPartnerModel,
    qwen       : QwenTextEncoder,
    clip       : CLIPTextEncoder,
    planner    : OnlinePlanner,
    global_text: str,
    init       : dict,
    device     : str,
    n_steps    : int,
    max_phases : int,
):
    def encode(text: str):
        ctxt = qwen.encode([text]).to(device)          # [1, T, 4096]
        vtxt = clip.encode([text]).to(device)          # [1, 1, 768]
        return ctxt, vtxt

    # ── World frames for the two actors ──────────────────────────────────────
    p1_origin = np.zeros(3, dtype=np.float32)
    p1_R      = np.eye(3, dtype=np.float32)

    p2_f0_in_p1ego = init["p2_f0_in_p1ego"].astype(np.float32)   # [201]
    p1_f0_in_p2ego = init["p1_f0_in_p2ego"].astype(np.float32)
    R2_full   = C.rot_from_6d(p2_f0_in_p1ego[3:9])
    yaw2      = float(np.arctan2(R2_full[0, 2], R2_full[2, 2]))
    p2_R      = p1_R @ C.yaw_rotation_matrix(yaw2)
    p2_origin = p1_R @ p2_f0_in_p1ego[0:3] + p1_origin
    p2_origin[1] = 0.0

    p1_hist: Optional[np.ndarray] = None
    p2_hist: Optional[np.ndarray] = None
    prev_p1_world: Optional[np.ndarray] = None
    prev_p2_world: Optional[np.ndarray] = None

    p1_list, p2_list, mask_list = [], [], []
    completed: List[CompletedPhase] = []
    elapsed = 0

    for pidx in range(max_phases):
        dist = float(np.linalg.norm(p1_origin[[0, 2]] - p2_origin[[0, 2]]))
        nxt  = planner.plan_next(global_text, completed, current_dist_m=dist, elapsed_frames=elapsed)
        if nxt is None:
            break

        W = int(nxt.bucket)
        logger.info("=== phase %d  type=%s bucket=%d dist=%.2f ===", pidx, nxt.phase_type, W, dist)

        ctxt1, vtxt1 = encode(nxt.p1_text)
        ctxt2, vtxt2 = encode(nxt.p2_text)

        # seq_mask_full = [10 history True | W generation True]   (online: no [N/A] pad)
        smf = torch.ones(1, W + OVERLAP_N, dtype=torch.bool, device=device)

        # ── self-history ─────────────────────────────────────────────────────
        if pidx == 0:
            hist1 = np.tile(init["p1_initial_frame"].astype(np.float32)[None], (OVERLAP_N, 1))
            hist2 = np.tile(init["p2_initial_frame"].astype(np.float32)[None], (OVERLAP_N, 1))
        else:
            hist1, hist2 = p1_hist, p2_hist
        hist1_n = model.normalize(torch.from_numpy(hist1).unsqueeze(0).to(device).float())
        hist2_n = model.normalize(torch.from_numpy(hist2).unsqueeze(0).to(device).float())

        # ── partner for P1 (Ping): P2's previous phase, in current P1 ego ─────
        if pidx == 0:
            p1_partner, p1_pmask = C.partner_to_tensor(p2_f0_in_p1ego[np.newaxis, :], device)
        else:
            p2_prev_in_p1ego = C.world_to_ego_motion(prev_p2_world, p1_origin, p1_R)
            p1_partner, p1_pmask = C.partner_to_tensor(p2_prev_in_p1ego, device)

        # ── Ping: P1 ─────────────────────────────────────────────────────────
        pred_p1 = C.sample_ode_030(model, ctxt1, vtxt1, smf, hist1_n, W,
                                   p1_partner, p1_pmask, n_steps=n_steps, device=device)[0].cpu().numpy()
        pred_p1 = HHIPartnerModel.smooth_motion(torch.from_numpy(pred_p1)).numpy()
        pred_p1_world = C.ego_to_world_motion(pred_p1, p1_origin, p1_R)

        # ── Pong: P2, partner = P1's just-generated phase, in current P2 ego ──
        p1_curr_in_p2ego = C.world_to_ego_motion(pred_p1_world, p2_origin, p2_R)
        p2_partner, p2_pmask = C.partner_to_tensor(p1_curr_in_p2ego, device)
        pred_p2 = C.sample_ode_030(model, ctxt2, vtxt2, smf, hist2_n, W,
                                   p2_partner, p2_pmask, n_steps=n_steps, device=device)[0].cpu().numpy()
        pred_p2 = HHIPartnerModel.smooth_motion(torch.from_numpy(pred_p2)).numpy()
        pred_p2_world = C.ego_to_world_motion(pred_p2, p2_origin, p2_R)

        # ── carry ego frame to next phase + extract self-history tail ─────────
        def _tail(pred_ego):
            t = pred_ego[-OVERLAP_N:].copy()
            if len(t) < OVERLAP_N:
                t = np.concatenate([np.tile(t[0:1], (OVERLAP_N - len(t), 1)), t])
            return t

        def _update_ego(pred_world, old_origin, old_R, tail_old_ego):
            last = pred_world[-1]
            new_origin = last[0:3].copy(); new_origin[1] = 0.0
            Rm  = C.rot_from_6d(last[3:9])
            yaw = float(np.arctan2(Rm[0, 2], Rm[2, 2]))
            new_R = C.yaw_rotation_matrix(yaw)
            tail_world   = C.ego_to_world_motion(tail_old_ego, old_origin, old_R)
            tail_new_ego = C.world_to_ego_motion(tail_world, new_origin, new_R)
            return new_origin, new_R, tail_new_ego

        prev_p1_world, prev_p2_world = pred_p1_world, pred_p2_world
        p1_origin, p1_R, p1_hist = _update_ego(pred_p1_world, p1_origin, p1_R, _tail(pred_p1))
        p2_origin, p2_R, p2_hist = _update_ego(pred_p2_world, p2_origin, p2_R, _tail(pred_p2))

        p1_list.append(pred_p1_world)
        p2_list.append(pred_p2_world)
        mask_list.append(np.ones(W, dtype=bool))

        completed.append(CompletedPhase(
            phase_idx=pidx, phase_type=nxt.phase_type, bucket=W,
            p1_text=nxt.p1_text, p2_text=nxt.p2_text, elapsed_frames=elapsed + W,
        ))
        elapsed += W

    if not p1_list:
        return None

    p1_world = HHIPartnerModel.smooth_motion(torch.from_numpy(np.concatenate(p1_list))).numpy()
    p2_world = HHIPartnerModel.smooth_motion(torch.from_numpy(np.concatenate(p2_list))).numpy()
    mask     = np.concatenate(mask_list)
    return p1_world, p2_world, mask, completed


# ── Entry ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_REPO, "checkpoints", "run_030", "best.pt"))
    ap.add_argument("--cfg_json", default=os.path.join(_REPO, "experiments", "train_030_config.json"))
    ap.add_argument("--norm_stats", default=os.path.join(_REPO, "data", "motion_norm_stats.npz"))
    ap.add_argument("--out_dir", default=os.path.join(_REPO, "outputs", "online"))
    # text / planner
    ap.add_argument("--demo_id", type=int, default=None, help="index into demo_global_texts.json")
    ap.add_argument("--global_text", default=None, help="custom goal (needs --allow_custom_global)")
    ap.add_argument("--allow_custom_global", action="store_true")
    ap.add_argument("--list", action="store_true", help="print demo list and exit")
    ap.add_argument("--llm_model", default="qwen3.5:35b")
    ap.add_argument("--llm_base_url", default=os.environ.get("LLM_BASE_URL", "http://localhost:11435/v1"))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_phases", type=int, default=8)
    # encoders / sampling
    ap.add_argument("--qwen_path", default=os.path.join(_HY, "ckpts", "Qwen3-8B"))
    ap.add_argument("--clip_path", default=os.path.join(_HY, "ckpts", "clip-vit-large-patch14"))
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--render", action="store_true", help="also write an HTML viewer")
    args = ap.parse_args()

    demo = load_demo_list()
    if args.list:
        for i, e in enumerate(demo):
            print(f"[{i:2d}] ({e['dataset']}) {e['seq_id']}: {e['global_text']}")
        return

    if args.seed is not None:
        import random
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    global_text = resolve_global_text(args, demo)

    model = C.load_run030_model(args.ckpt, args.cfg_json, args.norm_stats, args.device)
    qwen  = QwenTextEncoder(args.qwen_path, max_length=256, device=args.device)
    clip  = CLIPTextEncoder(args.clip_path, device=args.device)
    planner = OnlinePlanner(model_name=args.llm_model, base_url=args.llm_base_url,
                            temperature=args.temperature, max_phases=args.max_phases)

    result = run_episode(model, qwen, clip, planner, global_text, load_init(),
                         args.device, args.n_steps, args.max_phases)
    if result is None:
        logger.error("Episode produced no phases (planner returned nothing).")
        return
    p1_world, p2_world, mask, completed = result

    os.makedirs(args.out_dir, exist_ok=True)
    npz_path = os.path.join(args.out_dir, "motion.npz")
    np.savez(npz_path, p1=p1_world, p2=p2_world, mask=mask,
             global_text=global_text,
             phases=json.dumps([c.__dict__ for c in completed]))
    logger.info("Saved motion -> %s  (%d frames, %d phases)", npz_path, int(mask.sum()), len(completed))

    if args.render:
        out_html = os.path.join(args.out_dir, "vis.html")
        C.render_html(p1_world, p2_world, mask, global_text, out_html)


if __name__ == "__main__":
    main()
