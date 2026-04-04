"""Configuration constants for bias evaluation and baselines."""
import os

# Device: set to GPU index (e.g. 9) to use "cuda:9"; None = default (cuda:0 or cpu)
CUDA_DEVICE_ID = 9

# Model
QWEN_MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # Use small variant for experimentation; change to 0.5B-Instruct if needed
MAX_LENGTH = 512

# Paths (results written to D:\PoojaYakkala_Capstone_results\results when that folder exists)
_CHECKPOINT_DIR = "checkpoints"
_REPRESENTATIONS_DIR = "representations"
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXTERNAL_RESULTS = os.path.normpath(os.path.join(_PROJECT_ROOT, "..", "PoojaYakkala_Capstone_results", "results"))
CHECKPOINT_DIR = _CHECKPOINT_DIR
REPRESENTATIONS_DIR = _REPRESENTATIONS_DIR
RESULTS_DIR = _EXTERNAL_RESULTS if os.path.exists(os.path.dirname(_EXTERNAL_RESULTS)) else "results"

# Training defaults
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 3
DEFAULT_LR = 2e-5
DEFAULT_LORA_LR = 1e-4
DEFAULT_LORA_EPOCHS = 2

# INLP
DEFAULT_INLP_ITERATIONS = 10

# Probe
PROBE_C = 1.0  # sklearn LogisticRegression C
PROBE_MAX_ITER = 10000  # increased for convergence after scaling

# Baselines — adversarial λ1 and LoRA (stronger defaults: nonlinear bias head + scale; see models/adversarial.py)
DEFAULT_LAMBDA_BIAS = 1.0
DEFAULT_LAMBDA_STAB = 0.1   # Main model: stability term weight λ2
# LoRA capacity + dropout (try r=64, α=128; increase for more representational freedom)
DEFAULT_LORA_R = 64
DEFAULT_LORA_ALPHA = 128
DEFAULT_LORA_DROPOUT = 0.1
# When L_task >> L_bias, scale λ1*L_bias per batch so the adversary is not ignored (detached ratio).
DEFAULT_ADV_BIAS_LOSS_BALANCE = True
# Agentic inner loop + Main simulated adaptation (strong updates, L_task-only in eval when enabled)
DEFAULT_ADAPTATION_STEPS = 10
DEFAULT_ADAPTATION_LR = 1e-4
DEFAULT_MAIN_INNER_STEPS = 8
DEFAULT_MAIN_INNER_LR = 1e-4
DEFAULT_ALPHA_BIAS = 0.3   # B4 bias loss weight
DEFAULT_BETA_ORTH = 0.1    # B4 orthogonality weight

def get_device():
    """Return device string: cuda:CUDA_DEVICE_ID if set and CUDA available, else cuda:0 or cpu."""
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    n = torch.cuda.device_count()
    if CUDA_DEVICE_ID is not None:
        if CUDA_DEVICE_ID >= n:
            raise RuntimeError(
                f"config.CUDA_DEVICE_ID={CUDA_DEVICE_ID} but PyTorch only sees {n} GPU(s) "
                f"(indices 0–{n - 1}). If you set CUDA_VISIBLE_DEVICES=9, that GPU becomes index 0 here — "
                f"set CUDA_DEVICE_ID = 0 in config.py for that setup."
            )
        return f"cuda:{CUDA_DEVICE_ID}"
    return "cuda"


def log_device_banner(device: str) -> None:
    """Print one-line CUDA diagnostics after choosing ``device`` (call from runners)."""
    import torch
    print(f"Using device: {device}")
    if device.startswith("cuda"):
        idx = torch.device(device).index
        name = torch.cuda.get_device_name(idx)
        print(
            f"  CUDA: gpu_index={idx} | visible_count={torch.cuda.device_count()} | {name}"
        )
    else:
        if torch.cuda.is_available():
            print(
                f"  WARNING: PyTorch sees {torch.cuda.device_count()} GPU(s) but get_device() returned CPU — "
                "check config (e.g. CUDA_DEVICE_ID out of range was not hit if you forced cpu elsewhere)."
            )
        else:
            print(
                "  WARNING: torch.cuda.is_available() is False — install a CUDA build of PyTorch "
                "or drivers; training will be much slower on CPU."
            )

def ensure_dirs():
    for d in (CHECKPOINT_DIR, REPRESENTATIONS_DIR, RESULTS_DIR):
        os.makedirs(d, exist_ok=True)
