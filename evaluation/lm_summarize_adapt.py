"""
Causal-LM adaptation for **task shift**: instruction + biography → predict a short pseudo-summary.

Instruction (fixed): **"Summarize this biography."** — generation-style pressure on the backbone;
debiasing still used occupation classification. Gender evaluation stays a linear probe on pooled h.

Target text: first sentence or truncated words from the biography (no profession-description label).
"""
from __future__ import annotations

from typing import List, Tuple

import torch

LM_SUMMARY_INSTRUCTION = "Summarize this biography."


def pseudo_summary_from_bio_text(text: str, max_words: int = 48) -> str:
    """Cheap pseudo-target for LM loss: first sentence, else first ``max_words`` words."""
    t = (text or "").strip().replace("\n", " ")
    if not t:
        return "No biography text."
    for sep in ".?!":
        pos = t.find(sep)
        if 0 < pos <= min(400, len(t) - 1):
            sent = t[: pos + 1].strip()
            if len(sent) >= 12:
                return sent[:400]
    words = t.split()[:max_words]
    return " ".join(words) if words else t[:200]


def build_lm_summarize_batch_tensors(
    tokenizer,
    bio_texts: List[str],
    device: str,
    max_length: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build padded batch for Causal LM: [instruction] + bio + ``Summary:`` + pseudo-summary.
    Labels are -100 on the prefix (instruction + bio + ``Summary: ``) so loss is only on summary tokens.
    """
    rows_ids: List[torch.Tensor] = []
    rows_mask: List[torch.Tensor] = []
    rows_labels: List[torch.Tensor] = []

    for raw in bio_texts:
        bio = raw.strip()
        summary = pseudo_summary_from_bio_text(bio)
        full_text = f"{LM_SUMMARY_INSTRUCTION}\n\n{bio}\n\nSummary: {summary}"
        enc = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        ids = enc["input_ids"][0]
        mask = enc["attention_mask"][0]
        prefix = f"{LM_SUMMARY_INSTRUCTION}\n\n{bio}\n\nSummary: "
        pref_enc = tokenizer(
            prefix,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        n_pref = min(len(pref_enc["input_ids"]), int(ids.shape[0]))
        lab = ids.clone()
        lab[:n_pref] = -100
        # Mask padding positions if tokenizer added none (single seq)
        if (mask == 0).any():
            lab[mask == 0] = -100
        rows_ids.append(ids)
        rows_mask.append(mask)
        rows_labels.append(lab)

    max_t = max(int(x.shape[0]) for x in rows_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    b = len(rows_ids)
    batched_ids = torch.full((b, max_t), pad_id, dtype=torch.long)
    batched_mask = torch.zeros((b, max_t), dtype=torch.long)
    batched_labels = torch.full((b, max_t), -100, dtype=torch.long)
    for i, (ids, msk, lab) in enumerate(zip(rows_ids, rows_mask, rows_labels)):
        L = int(ids.shape[0])
        batched_ids[i, :L] = ids
        batched_mask[i, :L] = msk
        batched_labels[i, :L] = lab
        batched_labels[i, L:] = -100

    return (
        batched_ids.to(device),
        batched_mask.to(device),
        batched_labels.to(device),
    )


def backbone_lm_loss(backbone, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
    """Single forward; returns scalar loss (shifted CE inside HF CausalLM)."""
    out = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    return out.loss
