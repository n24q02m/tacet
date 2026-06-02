"""LLM teacher tier: the Tier-3 teacher protocol plus pluggable LLM adapters.

``teacher`` holds the offline oracle and the ``Teacher`` protocol; the
``teachers`` subpackage holds the real LLM adapters (Gemini, Grok, rotating
fallback chains) wired from settings. ``metering`` holds the opt-in real
token-cost meter (``MeteredTeacher`` / ``PriceTable``) for the real experiment.
"""

from tacet.llm.metering import MeteredTeacher, PriceTable

__all__ = ["MeteredTeacher", "PriceTable"]
