"""Production settings — env-driven configuration for the TACET service.

Reads ``TACET_*`` environment variables. Built on top of pydantic-settings
when available, falling back to a dataclass that reads ``os.environ``
directly — the framework runs in either case.

Typical use::

    from tacet.serve.settings import settings
    teacher = build_teacher(settings)            # picks Gemini or Grok by config
    server  = build_server(settings)             # FastAPI app with creds wired in
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Annotated

try:
    from pydantic import Field, field_validator
    from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover - optional dep
    _HAS_PYDANTIC = False


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None else default


if _HAS_PYDANTIC:

    class Settings(BaseSettings):
        """TACET service configuration (loaded from env or a ``.env`` file)."""

        model_config = SettingsConfigDict(
            env_prefix="TACET_",
            env_file=(".env", ".env.local"),
            extra="ignore",
        )

        # --- teacher selection ---------------------------------------------
        teacher: str = Field(
            default="oracle",
            description="oracle | gemini | grok | openrouter | fallback | rotating",
        )
        gemini_api_key: str | None = None
        gemini_model: str = "gemini-3.5-flash"
        gemini_endpoint: str = "generativelanguage"  # generativelanguage | vertex
        xai_api_key: str | None = None
        xai_model: str = "grok-4.3"
        xai_model_fast: str = "grok-4.3"
        xai_base_url: str = "https://api.x.ai/v1"
        # --- OpenRouter (OpenAI-compatible; first-class E11 ladder path) ----
        openrouter_api_key: str | None = None
        openrouter_model: str = "x-ai/grok-4.3"  # published anchor
        # --- free-tier rotation across Gemini / Gemma flash variants -------
        rotating_models: Annotated[list[str] | None, NoDecode] = (
            None  # None => DEFAULT_ROTATING_MODELS
        )
        rotating_cooldown_s: float = 60.0
        rotating_qps_per_model: float = 9 / 60  # ≤ 9 rpm per model

        # --- KGE backend ---------------------------------------------------
        kge_backend: str = Field(default="auto", description="auto | numpy | torch")
        kge_device: str = "auto"  # auto | cpu | cuda | mps
        kge_dim: int = 64
        kge_epochs: int = 200
        kge_model: str = Field(
            default="complex", description="complex (BCE) | complex_n3 (Lacroix) | rotate (Sun)"
        )

        # --- cascade -------------------------------------------------------
        l2_threshold: float = 0.6
        consolidate_every: int = 100
        synth_trigger: int = 10
        min_confidence: float = 0.95
        min_support: int = 3

        # --- server --------------------------------------------------------
        # Loopback by default so a bare `python -m tacet.serve.server` is not
        # exposed; the shipped Dockerfile sets TACET_HOST=0.0.0.0 to opt in.
        host: str = "127.0.0.1"
        port: int = 8088
        log_level: str = "INFO"
        # No CORS by default; an open "*" combined with no auth invites
        # cross-site abuse. Set TACET_CORS_ORIGINS (comma-separated) to opt in.
        cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
        # Optional API key for the mutating / cost-incurring endpoints
        # (/ask, /distill, /consolidate, /graph/edges). When unset the
        # endpoints are open (offline demo); set TACET_SERVER_API_KEY to require
        # a matching ``X-API-Key`` header.
        server_api_key: str | None = None

        # --- episodic / persistence ----------------------------------------
        episodes_path: str | None = None  # JSONL persistence; None = in-memory

        @field_validator("cors_origins", "rotating_models", mode="before")
        @classmethod
        def _split_csv_env(cls, v: object) -> object:
            """Accept a comma-separated env string for these list fields.

            ``NoDecode`` disables pydantic-settings' default JSON decoding of
            ``list`` env values, so a value like ``a.com,b.com`` reaches this
            validator as a raw string (JSON decoding would otherwise raise a
            ``SettingsError`` on the comma). Non-string values (the defaults,
            or values already parsed from a ``.env`` list) pass through.
            """
            if isinstance(v, str):
                return [item.strip() for item in v.split(",") if item.strip()]
            return v

        def has_real_teacher(self) -> bool:
            if self.teacher == "gemini":
                return bool(self.gemini_api_key)
            if self.teacher == "grok":
                return bool(self.xai_api_key)
            if self.teacher == "openrouter":
                return bool(self.openrouter_api_key)
            if self.teacher == "fallback":
                return bool(self.gemini_api_key) or bool(self.xai_api_key)
            if self.teacher == "rotating":
                return bool(self.gemini_api_key)
            return False

else:

    @dataclass
    class Settings:
        """TACET service configuration (env-driven dataclass fallback)."""

        teacher: str = "oracle"
        gemini_api_key: str | None = None
        gemini_model: str = "gemini-3.5-flash"
        gemini_endpoint: str = "generativelanguage"  # generativelanguage | vertex
        xai_api_key: str | None = None
        xai_model: str = "grok-4.3"
        xai_model_fast: str = "grok-4.3"
        xai_base_url: str = "https://api.x.ai/v1"
        openrouter_api_key: str | None = None
        openrouter_model: str = "x-ai/grok-4.3"  # published anchor
        rotating_models: list[str] | None = None
        rotating_cooldown_s: float = 60.0
        rotating_qps_per_model: float = 9 / 60
        kge_backend: str = "auto"
        kge_device: str = "auto"
        kge_dim: int = 64
        kge_epochs: int = 200
        kge_model: str = "complex"  # complex | complex_n3 | rotate
        l2_threshold: float = 0.6
        consolidate_every: int = 100
        synth_trigger: int = 10
        min_confidence: float = 0.95
        min_support: int = 3
        host: str = "127.0.0.1"
        port: int = 8088
        log_level: str = "INFO"
        cors_origins: list[str] = field(default_factory=list)
        server_api_key: str | None = None
        episodes_path: str | None = None

        @classmethod
        def from_env(cls) -> Settings:
            p = "TACET_"
            rotating_raw = _env(f"{p}ROTATING_MODELS")
            rotating = (
                [m.strip() for m in rotating_raw.split(",") if m.strip()] if rotating_raw else None
            )
            return cls(
                teacher=_env(f"{p}TEACHER", "oracle") or "oracle",
                gemini_api_key=_env(f"{p}GEMINI_API_KEY"),
                gemini_model=_env(f"{p}GEMINI_MODEL", "gemini-3.5-flash") or "gemini-3.5-flash",
                gemini_endpoint=_env(f"{p}GEMINI_ENDPOINT", "generativelanguage")
                or "generativelanguage",
                xai_api_key=_env(f"{p}XAI_API_KEY"),
                xai_model=_env(f"{p}XAI_MODEL", "grok-4.3") or "grok-4.3",
                xai_model_fast=_env(f"{p}XAI_MODEL_FAST", "grok-4.3") or "grok-4.3",
                xai_base_url=_env(f"{p}XAI_BASE_URL", "https://api.x.ai/v1")
                or "https://api.x.ai/v1",
                openrouter_api_key=_env(f"{p}OPENROUTER_API_KEY"),
                openrouter_model=_env(f"{p}OPENROUTER_MODEL", "x-ai/grok-4.3") or "x-ai/grok-4.3",
                rotating_models=rotating,
                rotating_cooldown_s=_env_float(f"{p}ROTATING_COOLDOWN_S", 60.0),
                rotating_qps_per_model=_env_float(f"{p}ROTATING_QPS_PER_MODEL", 9 / 60),
                kge_backend=_env(f"{p}KGE_BACKEND", "auto") or "auto",
                kge_device=_env(f"{p}KGE_DEVICE", "auto") or "auto",
                kge_dim=_env_int(f"{p}KGE_DIM", 64),
                kge_epochs=_env_int(f"{p}KGE_EPOCHS", 200),
                kge_model=_env(f"{p}KGE_MODEL", "complex") or "complex",
                l2_threshold=_env_float(f"{p}L2_THRESHOLD", 0.6),
                consolidate_every=_env_int(f"{p}CONSOLIDATE_EVERY", 100),
                synth_trigger=_env_int(f"{p}SYNTH_TRIGGER", 10),
                min_confidence=_env_float(f"{p}MIN_CONFIDENCE", 0.95),
                min_support=_env_int(f"{p}MIN_SUPPORT", 3),
                host=_env(f"{p}HOST", "127.0.0.1") or "127.0.0.1",
                port=_env_int(f"{p}PORT", 8088),
                log_level=_env(f"{p}LOG_LEVEL", "INFO") or "INFO",
                cors_origins=[
                    o.strip() for o in (_env(f"{p}CORS_ORIGINS", "") or "").split(",") if o.strip()
                ],
                server_api_key=_env(f"{p}SERVER_API_KEY"),
                episodes_path=_env(f"{p}EPISODES_PATH"),
            )

        def has_real_teacher(self) -> bool:
            if self.teacher == "gemini":
                return bool(self.gemini_api_key)
            if self.teacher == "grok":
                return bool(self.xai_api_key)
            if self.teacher == "openrouter":
                return bool(self.openrouter_api_key)
            if self.teacher == "fallback":
                return bool(self.gemini_api_key) or bool(self.xai_api_key)
            if self.teacher == "rotating":
                return bool(self.gemini_api_key)
            return False


def _backfill_provider_secrets(s: Settings) -> Settings:
    """Fall back to bare provider env vars when the TACET-scoped ones are
    absent.  This lets the service pick up a ``GEMINI_API_KEY`` /
    ``XAI_API_KEY`` already exported in the environment without setting
    any TACET-specific variables.
    """
    if not s.gemini_api_key:
        s.gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not s.xai_api_key:
        s.xai_api_key = os.environ.get("XAI_API_KEY")
    # Promote ``teacher=oracle`` to a real backend when a key turns up.
    # Prefer Grok (xAI) when both are present, otherwise fall back to the
    # Gemini free-tier rotation router.
    if s.teacher == "oracle":
        if s.xai_api_key:
            s.teacher = "grok"
        elif s.gemini_api_key:
            s.teacher = "rotating"
    return s


def load_settings() -> Settings:
    """Load settings from env (pydantic-settings if installed, else dataclass).

    After the primary load, also falls back to bare ``GEMINI_API_KEY`` /
    ``XAI_API_KEY`` so the service works without setting any TACET-specific
    env vars in production.
    """
    s = Settings() if _HAS_PYDANTIC else Settings.from_env()
    return _backfill_provider_secrets(s)


__all__ = ["Settings", "load_settings"]
