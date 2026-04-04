"""
Adversarial debiasing model: shared Qwen backbone, task head, bias head with GRL.
L_total = L_task + lambda * L_bias. Bias head predicts s; GRL makes backbone minimize bias.

Bias head is an MLP (not a single Linear): linear probes miss nonlinear structure; a stronger
adversary targets richer recoverability of the sensitive attribute.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Optional

from .grl import GradientReversalLayer

# When L_task >> L_bias, raw λ1*L_bias can be ignored; optional scaling brings terms closer (scale is detached).
ADV_BIAS_SCALE_MIN = 0.25
ADV_BIAS_SCALE_MAX = 4.0


def bias_loss_term(
    l_task: torch.Tensor,
    l_bias: torch.Tensor,
    lambda_bias: float,
    *,
    balance_magnitudes: bool = True,
) -> torch.Tensor:
    """
    Returns the adversarial term added to l_task (either λ1 * L_bias or λ1 * scale * L_bias).
    Gradients flow through l_bias; scale uses detached ratio so it acts as per-batch weighting.
    """
    if not balance_magnitudes:
        return lambda_bias * l_bias
    scale = (l_task.detach() / (l_bias.detach() + 1e-6)).clamp(ADV_BIAS_SCALE_MIN, ADV_BIAS_SCALE_MAX)
    return lambda_bias * scale * l_bias


class QwenAdversarialModel(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        num_task_labels: int = 2,
        num_bias_labels: int = 2,
        pooling: str = "mean",
        grl_alpha: float = 1.0,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModelForCausalLM.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size
        self.pooling = pooling
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        self.task_head = nn.Linear(hidden_size, num_task_labels)
        self.bias_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_bias_labels),
        )

    def _pool(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return hidden.mean(1)
        return hidden[:, -1, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        bias_labels: Optional[torch.Tensor] = None,
        return_hidden: bool = False,  # accepted for API parity with QwenTaskModel (hidden always returned)
    ) -> dict:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        h = self._pool(hidden, attention_mask)

        task_logits = self.task_head(h)
        h_rev = self.grl(h)
        bias_logits = self.bias_head(h_rev)

        out = {"logits": task_logits, "bias_logits": bias_logits, "hidden_states": h}
        loss_task = None
        if labels is not None:
            loss_task = nn.functional.cross_entropy(task_logits, labels)
            out["loss_task"] = loss_task
        loss_bias = None
        if bias_labels is not None:
            loss_bias = nn.functional.cross_entropy(bias_logits, bias_labels)
            out["loss_bias"] = loss_bias
        return out
