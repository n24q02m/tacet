"""Pluggable Tier-3 teachers — the offline oracle plus real LLM adapters."""

from tacet.llm.teachers.llm import (
    DEFAULT_ROTATING_MODELS,
    FallbackChainTeacher,
    GeminiRestTeacher,
    GeminiTeacher,
    GrokTeacher,
    RotatingTeacher,
    build_teacher_from_settings,
)

__all__ = [
    "DEFAULT_ROTATING_MODELS",
    "FallbackChainTeacher",
    "GeminiRestTeacher",
    "GeminiTeacher",
    "GrokTeacher",
    "RotatingTeacher",
    "build_teacher_from_settings",
]
