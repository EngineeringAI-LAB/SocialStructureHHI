"""
text_encoders.py
----------------
Online text encoders for the executor,  matching the OFFLINE encoding used
to build the training text-feature cache (see preprocess/encode_text_features.py).

  QwenTextEncoder : phase_text -> [T, 4096]  (Qwen last hidden state, chat template,
                    system-prefix cropped). T = max_length (256, matching cache).
  CLIPTextEncoder : phase_text -> [1, 768]   (CLIP-L pooler_output)

These must stay byte-for-byte consistent with the offline encoder, otherwise the
executor sees an out-of-distribution conditioning signal.
"""
from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# System prompt — identical to HunyuanMotion model_constants and the offline encoder.
_SYSTEM_PROMPT = (
    "Summarize human motion only from the user text for representation: "
    "action categories, key body-part movements, order/transitions, trajectory/direction, "
    "posture; include style/emotion/speed only if present. Explicitly capture laterality "
    "(left/right) when mentioned; do not guess. If multiple actions are described, indicate "
    "the count of distinct actions (e.g., actions=3) and their order. "
    "Do not invent missing info. Keep one concise paragraph."
)

TARGET_DIM = 4096   # backbone ctxt_input_dim; Qwen hidden_size == 4096 -> no projection


class QwenTextEncoder(nn.Module):
    """Qwen text encoder following HYTextModel.encode_llm (and the offline cache)."""

    def __init__(
        self,
        model_path : str,
        max_length : int = 256,           # matches the [256, 4096] training cache
        device     : str = "cuda",
        dtype      : torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading Qwen tokenizer from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        logger.info("Loading Qwen model from %s", model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, dtype=dtype,
        ).to(device).eval()
        self.model.requires_grad_(False)

        self.device     = device
        self.max_length = max_length

        cfg = self.model.config
        hidden_size = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size
        logger.info("Qwen hidden_size = %d, TARGET_DIM = %d", hidden_size, TARGET_DIM)
        if hidden_size != TARGET_DIM:
            self.proj = nn.Linear(hidden_size, TARGET_DIM, bias=False).to(device).float()
        else:
            self.proj = None

        self.crop_start = self._compute_crop_start()
        logger.info("crop_start = %d", self.crop_start)

    def _compute_crop_start(self) -> int:
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
        """Return [B, max_length, TARGET_DIM] float32 (cropped + optionally projected)."""
        full_max  = self.max_length + self.crop_start
        formatted = [self._apply_template(t) for t in texts]
        enc = self.tokenizer(
            formatted, return_tensors="pt", truncation=True,
            max_length=full_max, padding="max_length", return_attention_mask=True,
        )
        out = self.model(
            input_ids      = enc["input_ids"].to(self.device),
            attention_mask = enc["attention_mask"].to(self.device),
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1].float()
        start  = self.crop_start
        hidden = hidden[:, start:start + self.max_length].contiguous()
        if self.proj is not None:
            hidden = self.proj(hidden)
        return hidden   # [B, max_length, 4096] float32


class CLIPTextEncoder(nn.Module):
    """CLIP-L sentence embedding following HYTextModel.encode_sentence_emb."""

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
        """Return [B, 1, 768] float32 (pooler_output)."""
        enc = self.tokenizer(
            texts, return_tensors="pt", truncation=True,
            max_length=77, padding=True, return_attention_mask=True,
        )
        out = self.model(
            input_ids      = enc["input_ids"].to(self.device),
            attention_mask = enc["attention_mask"].to(self.device),
        )
        return out.pooler_output.float().unsqueeze(1)   # [B, 1, 768]
