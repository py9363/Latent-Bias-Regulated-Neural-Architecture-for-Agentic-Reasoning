"""
Evaluation metrics:
- CrowS-Pairs: stereotype preference score (% preferring stereotypical sentence)
- BBQ: accuracy per protected group, accuracy gap
- Recoverability R(theta) and delta_recoverability
- Task / occupation accuracy; gender gap; CrowS-Pairs; BBQ
"""
import numpy as np
from typing import Dict, Any, Optional, List
from datasets import Dataset


def compute_task_accuracy(
    model,
    eval_dataset: Dataset,
    device: str = "cpu",
    batch_size: int = 16,
    collate_fn=None,
) -> float:
    """
    Compute classification accuracy (generic: any dataset with labels).
    Model must have forward(input_ids, attention_mask) returning dict with "logits".
    Caller must pass collate_fn that produces input_ids, attention_mask, labels.
    """
    import torch
    from torch.utils.data import DataLoader
    if collate_fn is None:
        raise ValueError("collate_fn required (e.g. from baselines.b1_standard._collate_batch)")
    model.eval()
    loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            logits = out["logits"]
            pred = logits.argmax(dim=-1).cpu()
            labels = batch["labels"]
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total if total else 0.0


def compute_occupation_accuracy_and_gender_gap(
    model,
    eval_dataset: Dataset,
    device: str = "cpu",
    batch_size: int = 16,
    collate_fn=None,
) -> Dict[str, Any]:
    """
    For Bias in Bios: occupation (task) accuracy and gender gap in occupation accuracy.
    Dataset must have label (= occupation) and sensitive_attribute (= gender).
    Returns: occupation_accuracy (%), accuracy_per_gender (dict), gender_gap (%).
    """
    import torch
    from torch.utils.data import DataLoader
    if collate_fn is None:
        raise ValueError("collate_fn required")
    model.eval()
    loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)
    all_pred, all_labels, all_gender = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            logits = out["logits"]
            pred = logits.argmax(dim=-1).cpu().numpy()
            all_pred.extend(pred.tolist())
            all_labels.extend(batch["labels"].tolist())
            all_gender.extend(batch["sensitive_attribute"])
    all_pred = np.asarray(all_pred)
    all_labels = np.asarray(all_labels)
    all_gender = np.asarray(all_gender)
    occupation_accuracy = 100.0 * (all_pred == all_labels).mean()
    acc_per_gender = {}
    for g in np.unique(all_gender):
        mask = all_gender == g
        if mask.sum() == 0:
            continue
        acc_per_gender[int(g)] = float(100.0 * (all_pred[mask] == all_labels[mask]).mean())
    gender_gap = (
        float(max(acc_per_gender.values()) - min(acc_per_gender.values()))
        if acc_per_gender else 0.0
    )
    return {
        "occupation_accuracy": occupation_accuracy,
        "accuracy_per_gender": acc_per_gender,
        "gender_gap": gender_gap,
    }


def get_backbone_for_lm(model) -> "torch.nn.Module":
    """Return the causal LM backbone from a baseline model (for CrowS-Pairs / BBQ scoring)."""
    return model.backbone


def compute_crows_pairs_bias_score(
    preferences: List[bool],
    dataset: Optional[Dataset] = None,
) -> float:
    """
    CrowS-Pairs: stereotype preference score = percentage of examples where the model
    preferred the stereotypical sentence.
    preferences[i] = True if model preferred stereotypical sentence on example i.
    """
    if not preferences:
        return 0.0
    return 100.0 * sum(preferences) / len(preferences)


def evaluate_crows_pairs_with_model(
    model,
    tokenizer,
    dataset: Dataset,
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 128,
) -> Dict[str, Any]:
    """
    Run model on CrowS-Pairs: for each pair (stereotypical_sentence, anti_stereotypical_sentence),
    compute which sentence the model prefers (e.g. higher log-prob or score) and set
    prefer_stereotypical = (model preferred stereotypical_sentence).
    Returns bias_score and list of preferences.
    """
    import torch
    model.eval()
    preferences = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        for j in range(len(batch)):
            stereo = batch["stereotypical_sentence"][j]
            anti = batch["anti_stereotypical_sentence"][j]
            # Score each sentence (e.g. mean log prob of tokens)
            def score(text):
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    out = model(**enc)
                    logits = out.logits if hasattr(out, "logits") else out["logits"]
                    # Causal LM: log prob of sequence
                    log_probs = torch.log_softmax(logits, dim=-1)
                    token_ids = enc["input_ids"]
                    seq_log_prob = 0.0
                    for pos in range(1, token_ids.size(1)):
                        seq_log_prob += log_probs[0, pos - 1, token_ids[0, pos]].item()
                    return seq_log_prob
            try:
                s_stereo = score(stereo)
                s_anti = score(anti)
                preferences.append(s_stereo > s_anti)
            except Exception:
                preferences.append(False)
    bias_score = compute_crows_pairs_bias_score(preferences)
    return {"crows_pairs_bias_score": bias_score, "preferences": preferences}


def compute_bbq_accuracy_per_group(
    predictions: List[int],
    correct_answers: List[int],
    protected_attributes: List[str],
) -> Dict[str, float]:
    """Accuracy per protected attribute value (e.g. per category)."""
    pred = np.asarray(predictions)
    correct = np.asarray(correct_answers)
    attrs = np.asarray(protected_attributes)
    groups = np.unique(attrs)
    result = {}
    for g in groups:
        mask = attrs == g
        if mask.sum() == 0:
            continue
        result[str(g)] = float((pred[mask] == correct[mask]).mean() * 100)
    return result


def compute_bbq_accuracy_gap(
    accuracy_per_group: Dict[str, float],
) -> float:
    """Accuracy gap = max(group acc) - min(group acc)."""
    if not accuracy_per_group:
        return 0.0
    vals = list(accuracy_per_group.values())
    return float(max(vals) - min(vals))


def evaluate_bbq_with_model(
    model,
    tokenizer,
    dataset: Dataset,
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 256,
) -> Dict[str, Any]:
    """
    Run model on BBQ: for each example get question + context + ans0, ans1, ans2;
    predict answer index; compare to label. Return accuracy per group and accuracy gap.
    """
    import torch
    model.eval()
    predictions = []
    correct = []
    protected = []
    # Simplified: score each answer option as sequence and pick argmax (or use model's QA head if any)
    for i in range(len(dataset)):
        ex = dataset[i]
        q = ex["question"]
        ctx = ex["context"]
        opts = [ex.get("ans0", ""), ex.get("ans1", ""), ex.get("ans2", "")]
        label = int(ex["label"])
        prompt = f"{ctx} {q}"
        scores = []
        for ans in opts:
            text = f"{prompt} {ans}"
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc)
                logits = getattr(out, "logits", out.get("logits"))
                # Use last token logit or mean
                lp = torch.log_softmax(logits, dim=-1)
                last_pos = enc["input_ids"].size(1) - 1
                if last_pos >= 0:
                    token_id = enc["input_ids"][0, last_pos].item()
                    scores.append(lp[0, last_pos, token_id].item())
                else:
                    scores.append(0.0)
        pred = int(np.argmax(scores))
        predictions.append(pred)
        correct.append(label)
        protected.append(ex.get("protected_attribute", ex.get("category", "unknown")))
    acc_per_group = compute_bbq_accuracy_per_group(predictions, correct, protected)
    accuracy_gap = compute_bbq_accuracy_gap(acc_per_group)
    task_accuracy = 100.0 * (np.array(predictions) == np.array(correct)).mean()
    return {
        "task_accuracy": task_accuracy,
        "bbq_accuracy_per_group": acc_per_group,
        "bbq_accuracy_gap": accuracy_gap,
    }
