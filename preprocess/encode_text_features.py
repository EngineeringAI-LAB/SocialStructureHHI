"""
encode_text_features.py (v2)
-----------------------------
离线预编码文本特征，使用 Qwen3-8B + CLIP-L，仅编码 phase action text。

输出：
  phase_text_feat : [T, 4096]  Qwen3-8B last hidden state（chat template，crop system prefix）
  vtxt_feat       : [1, 768]   CLIP-L pooler_output（来自 phase_text）

保存路径：{out_dir}/{phase_stem}.npz
  dtype: float16（节省磁盘，加载时 cast to float32）

用法（单机多卡并行，每卡处理 1/N 分片）：
  python encode_text_features.py --data_dir data/dataset_npz --split train \\
      --qwen_path /path/to/Qwen3-8B --clip_path ckpts/clip-vit-large-patch14 \\
      --out_dir data/text_feat_cache_v2 --batch_size 32 \\
      --shard_id 0 --num_shards 4 --device cuda:0
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── System prompt（同 HunyuanMotion model_constants.py）─────────────────────
_SYSTEM_PROMPT = (
    "Summarize human motion only from the user text for representation: "
    "action categories, key body-part movements, order/transitions, trajectory/direction, "
    "posture; include style/emotion/speed only if present. Explicitly capture laterality "
    "(left/right) when mentioned; do not guess. If multiple actions are described, indicate "
    "the count of distinct actions (e.g., actions=3) and their order. "
    "Do not invent missing info. Keep one concise paragraph."
)

TARGET_DIM = 4096   # backbone ctxt_input_dim；Qwen3-8B hidden_size=4096，无需投影


# ── LLM encoder ──────────────────────────────────────────────────────────────

class QwenTextEncoder(nn.Module):
    """
    Qwen3.x 文本编码器（follow HYTextModel.encode_llm）。

    输出 last hidden state，crop 掉 system/user template prefix，
    投影至 TARGET_DIM（若 hidden_size 已经等于 TARGET_DIM 则 Identity）。
    """

    def __init__(
        self,
        model_path : str,
        max_length  : int = 256,
        device      : str = "cuda",
        dtype       : torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading Qwen tokenizer from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        logger.info("Loading Qwen model from %s", model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            dtype=dtype,
        ).to(device).eval()
        self.model.requires_grad_(False)

        self.device     = device
        self.max_length = max_length
        # Qwen3.5 是 VLM，hidden_size 嵌在 text_config 下
        cfg = self.model.config
        hidden_size = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size
        logger.info("Qwen hidden_size = %d, TARGET_DIM = %d", hidden_size, TARGET_DIM)

        if hidden_size != TARGET_DIM:
            logger.info("Adding linear projection %d → %d", hidden_size, TARGET_DIM)
            self.proj = nn.Linear(hidden_size, TARGET_DIM, bias=False).to(device).to(torch.float32)
        else:
            self.proj = None

        # 计算 system/user template prefix 长度（crop_start），同 HYTextModel
        self.crop_start = self._compute_crop_start()
        logger.info("crop_start = %d", self.crop_start)

    def _compute_crop_start(self) -> int:
        """找到 <BOC> marker 在模板编码后的起始 token 位置。"""
        marker = "<BOC>"
        msgs = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": marker},
        ]
        s = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        full_ids   = self.tokenizer(s, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
        marker_ids = self.tokenizer(marker, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        for i in range(len(full_ids) - len(marker_ids) + 1):
            if full_ids[i:i + len(marker_ids)] == marker_ids:
                return i
        return max(0, len(full_ids) - 1)

    def _apply_template(self, text: str) -> str:
        msgs = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """返回 [B, max_length, TARGET_DIM] float32（已 crop + project）。"""
        full_max = self.max_length + self.crop_start
        formatted = [self._apply_template(t) for t in texts]
        enc = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=full_max,
            padding="max_length",
            return_attention_mask=True,
        )
        out = self.model(
            input_ids      = enc["input_ids"].to(self.device),
            attention_mask = enc["attention_mask"].to(self.device),
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1].float()              # [B, L, hidden_size]
        start  = self.crop_start
        end    = start + self.max_length
        hidden = hidden[:, start:end].contiguous()          # [B, max_length, hidden_size]
        if self.proj is not None:
            hidden = self.proj(hidden)                      # [B, max_length, TARGET_DIM]
        return hidden   # float32


# ── CLIP encoder ──────────────────────────────────────────────────────────────

class CLIPTextEncoder(nn.Module):
    """CLIP-L sentence embedding（follow HYTextModel.encode_sentence_emb）。"""

    def __init__(self, model_path: str, device: str = "cuda") -> None:
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer

        logger.info("Loading CLIP from %s", model_path)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, max_length=77)
        self.model     = CLIPTextModel.from_pretrained(model_path).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """返回 [B, 1, 768] float32（pooler_output）。"""
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=77,
            padding=True,
            return_attention_mask=True,
        )
        out    = self.model(
            input_ids      = enc["input_ids"].to(self.device),
            attention_mask = enc["attention_mask"].to(self.device),
        )
        vtxt = out.pooler_output.float().unsqueeze(1)   # [B, 1, 768]
        return vtxt


# ── Global text lookup ────────────────────────────────────────────────────────

def build_global_text_lookup(llm_cache_dir: str) -> dict:
    lookup = {}
    for path in glob.glob(os.path.join(llm_cache_dir, "*.json")):
        try:
            with open(path) as f:
                d = json.load(f)
            seq_id = d.get("seq_id", "")
            text   = d.get("global_text", "")
            if seq_id and text:
                lookup[seq_id] = text
        except Exception:
            pass
    logger.info("global_text lookup: %d sequences", len(lookup))
    return lookup


# ── Main encoding loop ────────────────────────────────────────────────────────

def encode_all(
    data_dir   : str,
    split      : str,
    out_dir    : str,
    qwen_path  : str,
    clip_path  : str,
    max_length : int,
    batch_size : int,
    device     : str,
    shard_id   : int,
    num_shards : int,
    llm_cache_dir: str = "",  # unused, kept for backward compat
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ── 收集所有 npz 文件 ────────────────────────────────────────────────────
    all_files = sorted(glob.glob(os.path.join(data_dir, split, "**", "*.npz"), recursive=True))
    # 分片
    all_files = all_files[shard_id::num_shards]
    logger.info("Shard %d/%d: %d files", shard_id, num_shards, len(all_files))

    # 跳过已处理
    pending = []
    for path in all_files:
        phase_id = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, f"{phase_id}.npz")
        if not os.path.exists(out_path):
            pending.append(path)
    logger.info("%d files pending (skipping %d already done)", len(pending), len(all_files) - len(pending))
    if not pending:
        return

    # ── 加载模型 ─────────────────────────────────────────────────────────────
    qwen = QwenTextEncoder(qwen_path, max_length=max_length, device=device)
    clip = CLIPTextEncoder(clip_path, device=device)

    # ── 批处理 ───────────────────────────────────────────────────────────────
    for batch_start in range(0, len(pending), batch_size):
        batch_paths = pending[batch_start : batch_start + batch_size]

        phase_texts: List[str] = []
        file_stems : List[str] = []   # npz stem（含 _P1/_P2），用作输出文件名

        for path in batch_paths:
            data      = np.load(path, allow_pickle=True)
            file_stem = os.path.splitext(os.path.basename(path))[0]   # e.g. IH_1011_phase000_P1
            p_text    = str(data["phase_text"].flat[0]) if "phase_text" in data.files else ""
            phase_texts.append(p_text)
            file_stems.append(file_stem)

        # 仅编码 phase_text；vtxt 也来自 phase_text（不再使用 global_text）
        phase_feat = qwen.encode(phase_texts)   # [B, T, 4096]
        vtxt_feat  = clip.encode(phase_texts)   # [B, 1, 768]

        # 逐样本存盘（float16 节省空间）
        for i, file_stem in enumerate(file_stems):
            out_path = os.path.join(out_dir, f"{file_stem}.npz")
            np.savez_compressed(
                out_path,
                phase_text_feat = phase_feat[i].cpu().to(torch.float16).numpy(),  # [T, 4096]
                vtxt_feat       = vtxt_feat[i].cpu().to(torch.float16).numpy(),   # [1, 768]
            )

        logger.info(
            "[%d/%d] encoded batch %d–%d",
            batch_start + len(batch_paths), len(pending),
            batch_start, batch_start + len(batch_paths) - 1,
        )

    logger.info("Done. Output in %s", out_dir)


# ── Entry ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",      default="data/dataset_npz")
    parser.add_argument("--split",         default="train", choices=["train", "val", "test"])
    parser.add_argument("--out_dir",       default="data/text_feat_cache")
    parser.add_argument("--qwen_path",
                        default="/scratch3/wan451/3DBody/HY-Motion-1.0/ckpts/Qwen3-8B",
                        help="Path to Qwen3-8B model directory")
    parser.add_argument("--clip_path",
                        default="/scratch3/wan451/3DBody/HY-Motion-1.0/ckpts/clip-vit-large-patch14")
    parser.add_argument("--llm_cache_dir", default="data/llm_annot_cache")
    parser.add_argument("--max_length",    type=int, default=128,
                        help="Max Qwen output token length after crop")
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--device",        default="cuda:0")
    parser.add_argument("--shard_id",      type=int, default=0)
    parser.add_argument("--num_shards",    type=int, default=1)
    args = parser.parse_args()

    # 路径相对于 HHI_GEN 项目根目录
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def _abs(p): return p if os.path.isabs(p) else os.path.join(root, p)

    encode_all(
        data_dir      = _abs(args.data_dir),
        split         = args.split,
        out_dir       = _abs(args.out_dir),
        qwen_path     = _abs(args.qwen_path),
        clip_path     = _abs(args.clip_path),
        llm_cache_dir = _abs(args.llm_cache_dir),
        max_length    = args.max_length,
        batch_size    = args.batch_size,
        device        = args.device,
        shard_id      = args.shard_id,
        num_shards    = args.num_shards,
    )


if __name__ == "__main__":
    main()
