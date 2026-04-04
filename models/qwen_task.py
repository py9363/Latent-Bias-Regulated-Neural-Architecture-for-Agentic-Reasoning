"""Qwen backbone + task head for standard fine-tuning (B1) and INLP (B3)."""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Optional, Tuple


class QwenTaskModel(nn.Module):
    """
    Qwen backbone with a task classification head.
    Used for B1 (standard) and B3 (INLP: task head on projected h).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        num_labels: int = 2,
        hidden_size: Optional[int] = None,
        pooling: str = "mean",  # "mean" or "last" (last token / CLS)
        projection_matrix: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModelForCausalLM.from_pretrained(model_name, config=self.config)
        self.hidden_size = hidden_size or self.config.hidden_size
        self.pooling = pooling
        self.num_labels = num_labels
        self.projection_matrix = projection_matrix  # for INLP: P @ h

        self.task_head = nn.Linear(self.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> dict:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]  # (B, seq, hidden_size)

        if self.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                h = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            else:
                h = hidden.mean(1)
        else:
            h = hidden[:, -1, :]

        if self.projection_matrix is not None:
            # INLP: h_clean = P @ h
            P = self.projection_matrix.to(h.device)
            h = torch.matmul(h, P.T)

        logits = self.task_head(h)

        out = {"logits": logits, "hidden_states": h}
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            out["loss"] = loss_fct(logits, labels)
        return out
