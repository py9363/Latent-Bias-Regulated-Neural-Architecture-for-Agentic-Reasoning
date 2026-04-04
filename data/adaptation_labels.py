"""
Adaptation objectives for TABLE 0 and agentic inner loop (Task B vs debiasing Task A).

Default **summarize_lm**: causal LM on ``"Summarize this biography."`` + bio + pseudo-summary target
(see ``evaluation/lm_summarize_adapt.py``). Gender probe on pooled states is unchanged.

**occupation**: same 28-way head as debiasing (often little drift).
"""
from __future__ import annotations

ADAPT_OBJECTIVE_OCCUPATION = "occupation"
ADAPT_OBJECTIVE_SUMMARIZE_LM = "summarize_lm"
