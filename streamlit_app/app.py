"""
Twi ↔ Chinese Neural Machine Translation — Streamlit app.
Run with:  streamlit run app.py
"""

import base64
import os
import sys
import warnings

import streamlit as st

warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*Examining the path.*')

sys.path.insert(0, os.path.dirname(__file__))

from app_utils.model_loader import load_engine
from components.translator import TranslationInterface
from components.sidebar import Sidebar
from components.history import TranslationHistory

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Twi ⇄ Chinese Translator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
_APP_DIR = os.path.dirname(__file__)
css_path = os.path.join(_APP_DIR, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── session state defaults ────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


def _b64_img(path: str) -> str:
    """Return a base64 data-URI for an image file."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    return f"data:image/{mime};base64,{data}"


def _flag_banner() -> str:
    """Build the side-by-side flag banner HTML."""
    img_dir = os.path.join(_APP_DIR, "images")
    ghana_path  = os.path.join(img_dir, "Ghana-Flag.jpg")
    china_path  = os.path.join(img_dir, "chinese_flag.jpg")

    if not (os.path.exists(ghana_path) and os.path.exists(china_path)):
        return ""

    ghana_uri = _b64_img(ghana_path)
    china_uri = _b64_img(china_path)

    return f"""
<div class="flag-banner">
  <div class="flag-card">
    <img src="{ghana_uri}" alt="Ghana Flag" />
    <span class="flag-label">Ghana</span>
  </div>
  <div class="flag-divider">⇄</div>
  <div class="flag-card">
    <img src="{china_uri}" alt="China Flag" />
    <span class="flag-label">China</span>
  </div>
</div>
"""


def main():
    # ── copyright bar ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="copyright-bar">&copy; trudey@uestc2025</div>',
        unsafe_allow_html=True,
    )

    # ── flag banner ───────────────────────────────────────────────────────────
    banner = _flag_banner()
    if banner:
        st.markdown(banner, unsafe_allow_html=True)

    # ── header ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="main-header">
            <h1>Twi ⇄ Chinese Translator</h1>
            <p>Bidirectional Neural Machine Translation powered by Transformer</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # sidebar (returns config dict)
    config = Sidebar().render()

    # pre-load model (cached — runs only once per server session)
    try:
        load_engine()
    except FileNotFoundError as exc:
        st.error(f"❌ {exc}")
        st.info(
            "Place the three model files in the  `models/`  directory "
            "(or set the  `MODEL_DIR`  environment variable to their location)."
        )
        return

    # main translation UI
    translator = TranslationInterface(config)
    result = translator.render()

    # save to history
    if result:
        st.session_state.history.insert(0, result)
        st.session_state.history = st.session_state.history[:50]

    # history panel
    TranslationHistory(st.session_state.history).render()

    # footer
    st.markdown(
        """
        <div class="footer">
            <p>Transformer &middot; SentencePiece BPE &middot; Twi ↔ Chinese</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
