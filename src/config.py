"""Credential + config access that works locally (.env) and on Streamlit
Community Cloud (``st.secrets``).

Look-up order: process environment (populated from ``.env`` locally) →
Streamlit secrets (populated on the cloud) → the supplied default.
"""

import os

_dotenv_loaded = False


def _ensure_dotenv():
    """Populate ``os.environ`` from a local ``.env`` (idempotent, best-effort)."""
    global _dotenv_loaded
    if not _dotenv_loaded:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass
        _dotenv_loaded = True


def get_secret(key: str, default=None):
    """Return the value of ``key`` from the env, then ``st.secrets``."""
    _ensure_dotenv()
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st

        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        # streamlit not installed / not in a streamlit runtime context.
        pass
    return default
