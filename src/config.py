from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL = "gemini-3.1-flash-lite"


@dataclass(frozen=True)
class Settings:
    root: Path = _ROOT
    data_path: Path = _ROOT / "data" / "cortex.parquet"

    # All relative dates ("this year", "this quarter") resolve against this,
    # not the wall clock — the ledger is a closed book ending 2025-03-31.
    reporting_as_of: _dt.date = _dt.date(2025, 3, 31)

    max_sub_questions: int = 5
    fuzzy_threshold: int = 82  # rapidfuzz WRatio 0-100; below this = unresolved

    llm_model_fast: str = os.getenv("LLM_MODEL_FAST", _DEFAULT_MODEL)
    llm_model_smart: str = os.getenv("LLM_MODEL_SMART", _DEFAULT_MODEL)
    llm_temperature: float = 0.0

    @property
    def google_api_key(self) -> str | None:
        return os.getenv("GOOGLE_API_KEY")


settings = Settings()


def get_llm(tier: str = "fast", **overrides):
    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set — add it to .env (see .env.example)."
        )
    model = settings.llm_model_smart if tier == "smart" else settings.llm_model_fast
    params = {
        "model": model,
        "google_api_key": settings.google_api_key,
        "temperature": settings.llm_temperature,
    }
    params.update(overrides)
    return ChatGoogleGenerativeAI(**params)
