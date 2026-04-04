"""
LoRA adaptation: apply PEFT LoRA to Qwen attention layers, freeze base, train 1–3 epochs
on instruction-following data. Save theta', extract hidden states, re-run probe -> delta_R.
"""
import os
import torch
from pathlib import Path
from typing import Optional, Tuple
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CHECKPOINT_DIR, REPRESENTATIONS_DIR, DEFAULT_LORA_LR, DEFAULT_LORA_EPOCHS, ensure_dirs, get_device
from evaluation.probe import R_theta, run_probe


def _get_lora_target_modules(model_name: str = "Qwen/Qwen2.5-0.5B"):
    """Qwen2 attention layers: q_proj, k_proj, v_proj, o_proj."""
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def run_lora_adaptation(
    base_model_name: str = "Qwen/Qwen2.5-0.5B",
    adaptation_dataset: Optional[Dataset] = None,
    instruction_dataset: Optional[Dataset] = None,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    epochs: int = DEFAULT_LORA_EPOCHS,
    lr: float = DEFAULT_LORA_LR,
    batch_size: int = 4,
    max_length: int = 256,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, str]:
    """
    Apply LoRA to Qwen attention layers, freeze base, train on instruction data.
    Returns (adapted_model, checkpoint_path).
    """
    from peft import get_peft_model, LoraConfig, TaskType

    ensure_dirs()
    output_dir = output_dir or os.path.join(CHECKPOINT_DIR, "lora_adapted")
    os.makedirs(output_dir, exist_ok=True)
    device = device or get_device()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
    target_modules = _get_lora_target_modules(base_model_name)
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model = model.to(device)

    # Build simple instruction dataset if not provided
    if instruction_dataset is None and adaptation_dataset is None:
        # Minimal dummy instruction pairs for short adaptation
        from datasets import Dataset as HFDataset
        instruction_dataset = HFDataset.from_dict({
            "input_ids": [[0] * min(64, max_length)] * 32,
            "attention_mask": [[1] * min(64, max_length)] * 32,
            "labels": [[0] * min(64, max_length)] * 32,
        })
        # Use adaptation_dataset for actual data if provided
    ds = instruction_dataset if instruction_dataset is not None else adaptation_dataset
    if ds is None:
        ds = instruction_dataset

    def _collate(batch):
        keys = list(batch[0].keys())
        out = {}
        for k in keys:
            if k in ("input_ids", "attention_mask", "labels") and isinstance(batch[0][k], (list, tuple)):
                out[k] = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(b[k][:max_length]) for b in batch],
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id or 0,
                )
            elif k == "labels":
                out[k] = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(b[k][:max_length]) for b in batch],
                    batch_first=True,
                    padding_value=-100,
                )
            else:
                out[k] = torch.stack([torch.tensor(b[k]) for b in batch])
        return out

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device)
                labels[labels == tokenizer.pad_token_id] = -100
            else:
                labels = input_ids.clone()
                labels[labels == tokenizer.pad_token_id] = -100
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, output_dir


def extract_hidden_after_lora(
    model,
    tokenizer,
    dataset: Dataset,
    batch_size: int = 8,
    max_length: int = 256,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, list]:
    """Extract last-layer mean-pooled hidden states from LoRA-adapted model."""
    device = device or next(model.parameters()).device
    model.eval()
    all_h = []
    all_s = []
    from baselines.b1_standard import _collate_batch
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_batch)
    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                output_hidden_states=True,
            )
            hidden = out.hidden_states[-1]
            mask = batch["attention_mask"].to(device).unsqueeze(-1).float()
            h = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_h.append(h.cpu())
            all_s.extend(batch["sensitive_attribute"])
    return torch.cat(all_h, dim=0), all_s


def compute_delta_R(
    R_theta_baseline: float,
    representations_path_adapted: Optional[str] = None,
    hidden_states_adapted: Optional[torch.Tensor] = None,
    sensitive_attributes_adapted: Optional[list] = None,
    **probe_kwargs,
) -> float:
    """
    delta_R = R(theta') - R(theta).
    R(theta') is computed from adapted representations (path or arrays).
    """
    if representations_path_adapted and os.path.isfile(representations_path_adapted):
        R_adapted = R_theta(representations_path=representations_path_adapted, **probe_kwargs)
    elif hidden_states_adapted is not None and sensitive_attributes_adapted is not None:
        R_adapted = R_theta(hidden_states=hidden_states_adapted, sensitive_attributes=sensitive_attributes_adapted, **probe_kwargs)
    else:
        raise ValueError("Provide representations_path_adapted or (hidden_states_adapted, sensitive_attributes_adapted)")
    return R_adapted - R_theta_baseline


def run_lora_from_baseline_checkpoint(
    baseline_checkpoint_path: Optional[str] = None,
    base_model_name: str = "Qwen/Qwen2.5-0.5B",
    adaptation_dataset: Optional[Dataset] = None,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    epochs: int = DEFAULT_LORA_EPOCHS,
    lr: float = DEFAULT_LORA_LR,
    batch_size: int = 4,
    max_length: int = 256,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
    baseline_name: str = "b1",
) -> Tuple[torch.nn.Module, str]:
    """
    Load baseline backbone from checkpoint, apply LoRA, adapt on dataset.
    Returns (adapted_model, output_dir). Use extract_hidden_after_lora(adapted_model, ...) then probe for R(θ').
    """
    from peft import get_peft_model, LoraConfig, TaskType

    ensure_dirs()
    output_dir = output_dir or os.path.join(CHECKPOINT_DIR, f"lora_from_{baseline_name}")
    os.makedirs(output_dir, exist_ok=True)
    device = device or get_device()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)

    # Load backbone weights from baseline checkpoint if provided (B0 pretrained uses no checkpoint)
    if baseline_checkpoint_path is not None:
        ckpt = torch.load(baseline_checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        backbone_state = {k.replace("backbone.", ""): v for k, v in state.items() if k.startswith("backbone.")}
        if backbone_state:
            model.load_state_dict(backbone_state, strict=False)
    model = model.to(device)

    target_modules = _get_lora_target_modules(base_model_name)
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model = model.to(device)

    if adaptation_dataset is None:
        from datasets import Dataset as HFDataset
        adaptation_dataset = HFDataset.from_dict({
            "input_ids": [[0] * min(64, max_length)] * 32,
            "attention_mask": [[1] * min(64, max_length)] * 32,
            "labels": [[0] * min(64, max_length)] * 32,
        })
    ds = adaptation_dataset

    def _collate_lora(batch):
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(b["input_ids"][:max_length]) for b in batch],
            batch_first=True,
            padding_value=pad_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(b["attention_mask"][:max_length]) for b in batch],
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(b["labels"][:max_length]) for b in batch],
            batch_first=True,
            padding_value=-100,
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_lora)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device)
                labels[labels == tokenizer.pad_token_id] = -100
            else:
                labels = input_ids.clone()
                labels[labels == tokenizer.pad_token_id] = -100
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, output_dir
