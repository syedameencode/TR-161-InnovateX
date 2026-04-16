"""
explainer.py
============
AI Security Analyst — Gemini 1.5 Flash integration for SentinelAI.

Provides a single cached function:
    explain_anomaly(log_line: str) -> str

The function calls the Gemini API with a Senior SOC Analyst system
instruction and returns a concise threat explanation + mitigation step.

Usage:
    from explainer import explain_anomaly
    insight = explain_anomaly(log_line_text)
"""

import os
import logging

import streamlit as st

# ── Load .env before anything else ────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — rely on environment variable directly

logger = logging.getLogger(__name__)

# ── System prompt for the SOC Analyst persona ─────────────────────────────────
_SYSTEM_INSTRUCTION = (
    "You are a Senior SOC (Security Operations Center) Analyst with 10+ years of "
    "experience in threat detection and incident response. "
    "Analyze the server log line provided, which has been flagged as an anomaly by "
    "an ML ensemble (Isolation Forest + One-Class SVM). "
    "Respond in exactly two clearly labelled sections:\n\n"
    "**🔍 Threat Analysis:** A concise (2-3 sentence) explanation of why this log "
    "entry is suspicious and what attack vector or system failure it indicates.\n\n"
    "**✅ Mitigation Step:** One specific, actionable remediation step an engineer "
    "can take immediately to contain or fix the issue.\n\n"
    "Be direct and technical. Avoid preamble."
)

# ── Lazy client initialisation ─────────────────────────────────────────────────
_client = None


def _get_client():
    """Lazily initialise the Gemini client, raising a clear error if key missing."""
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file or set it as an environment variable."
        )

    try:
        from google import genai  # google-genai SDK
        _client = genai.Client(api_key=api_key)
    except ImportError as exc:
        raise ImportError(
            "google-genai package is not installed. "
            "Run: pip install google-genai"
        ) from exc

    return _client


# ── Cached explainer function ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def explain_anomaly(log_line: str) -> str:
    """
    Call Gemini 1.5 Flash to explain why `log_line` is flagged as an anomaly
    and provide one actionable mitigation step.

    Results are cached by Streamlit for 1 hour so repeated calls for the
    same log line don't consume API quota.

    Parameters
    ----------
    log_line : str
        The raw log line text to analyse.

    Returns
    -------
    str
        Markdown-formatted explanation from Gemini, or an error message.
    """
    if not log_line or not log_line.strip():
        return "_No log content provided._"

    # Truncate extremely long lines to avoid token waste
    truncated = log_line.strip()[:1200]

    try:
        from google.genai import types

        client = _get_client()

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=truncated,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                temperature=0.3,      # low temperature → deterministic, factual
                max_output_tokens=400,
            ),
        )

        result = response.text.strip() if response.text else "_No response from model._"
        logger.info("Gemini explain_anomaly: success (%d chars)", len(result))
        return result

    except EnvironmentError as exc:
        logger.error("Gemini key error: %s", exc)
        return f"⚠️ **Configuration Error:** {exc}"
    except Exception as exc:  # noqa: BLE001
        logger.error("Gemini API error: %s", exc)
        return f"⚠️ **API Error:** {exc}"
