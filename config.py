import os
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
VISUALIZATIONS_DIR = BASE_DIR / "visualizations"
REPORT_DIR = BASE_DIR / "report"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("MODEL", "gemini-2.0-flash-lite")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", 3))
SEED = int(os.getenv("SEED", 42))

CHUNK_SIZE_TOKENS = 200
CHUNK_OVERLAP_TOKENS = 50
TOP_K_RETRIEVAL = 3
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 192))
CONFIDENCE_THRESHOLD = 0.7
BOOTSTRAP_RESAMPLES = 1000
SIMILARITY_THRESHOLD = 0.75
REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", 15))
RATE_LIMIT_COOLDOWN_SECONDS = int(os.getenv("RATE_LIMIT_COOLDOWN_SECONDS", 10))

REFUSAL_PHRASES = [
    "not present",
    "not in context",
    "i don't know",
    "cannot find",
    "not available",
    "no information",
    "not mentioned",
    "unable to find",
    "insufficient context",
    "cannot determine",
]


def ensure_directories() -> None:
    for directory in (DATA_DIR, RESULTS_DIR, VISUALIZATIONS_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)



def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass



def has_live_api_key(api_key: str | None = None) -> bool:
    value = GEMINI_API_KEY if api_key is None else api_key
    return bool(value and value != "your_key_here")



def validate_runtime_config() -> None:
    if not has_live_api_key():
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Update .env with a real key before running the benchmark."
        )
