"""Main translation interface component."""

import os
import time
import streamlit as st
from app_utils.model_loader import translate

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _load_examples(filename: str) -> list[str]:
    path = os.path.join(_DATA_DIR, filename)
    try:
        with open(path, encoding="utf-8") as f:
            return [l.rstrip("\n") for l in f if l.strip()]
    except FileNotFoundError:
        return []


# ── confidence bar renderer ───────────────────────────────────────────────────
def _confidence_bar(confidence: float) -> str:
    pct = confidence * 100
    if pct >= 68:
        gradient = "linear-gradient(90deg, #28a745 0%, #20c997 100%)"
        badge, badge_cls = "High", "high"
    elif pct >= 42:
        gradient = "linear-gradient(90deg, #ffc107 0%, #fd7e14 100%)"
        badge, badge_cls = "Medium", "medium"
    else:
        gradient = "linear-gradient(90deg, #dc3545 0%, #e83e8c 100%)"
        badge, badge_cls = "Low", "low"

    return f"""
<div class="conf-wrap">
  <div class="conf-header">
    <span class="conf-title">Confidence Score</span>
    <span class="conf-badge conf-{badge_cls}">{badge}</span>
    <span class="conf-pct">{pct:.1f}%</span>
  </div>
  <div class="conf-track">
    <div class="conf-fill"
         style="width:{pct:.1f}%; background:{gradient};"></div>
  </div>
</div>
"""

# ── language detection ────────────────────────────────────────────────────────
def _has_chinese(text: str) -> bool:
    return any(
        '\u4e00' <= c <= '\u9fff'
        or '\u3400' <= c <= '\u4dbf'
        or '\uf900' <= c <= '\ufaff'
        or '\u3000' <= c <= '\u303f'
        for c in text
    )

def _lang_ok(text: str, source_lang: str) -> bool:
    if not text.strip():
        return True
    return _has_chinese(text) if source_lang == "Chinese" else not _has_chinese(text)


_NONE = "— select an example —"

# Loaded once at import time from bundled data files (500 sentences each)
_EXAMPLES = {
    "Twi":     _load_examples("examples_twi.txt"),
    "Chinese": _load_examples("examples_chi.txt"),
}


class TranslationInterface:
    """Renders the main translation interface."""

    def __init__(self, config: dict):
        self.config = config

    def render(self):
        direction = self.config["direction"]

        if direction == "Twi → Chinese":
            source_lang, target_lang, model_dir = "Twi", "Chinese", "twi2chi"
            placeholder = "Type or paste Twi text here…"
        else:
            source_lang, target_lang, model_dir = "Chinese", "Twi", "chi2twi"
            placeholder = "请在此输入中文…"

        # Clear inputs when direction changes
        if st.session_state.get("_direction") != direction:
            st.session_state["source_text"]       = ""
            st.session_state["translation_output"] = ""
            st.session_state["_direction"]         = direction

        col1, col2 = st.columns(2)

        # ── source panel ──────────────────────────────────────────────────────
        with col1:
            st.markdown(f"**{source_lang}**")

            examples = _EXAMPLES.get(source_lang, [])
            # Use language-specific key so selectbox resets on direction change
            example_choice = st.selectbox(
                "Example sentences",
                [_NONE] + examples,
                key=f"example_{source_lang}",
            )

            # ── KEY FIX: push selected example into session state BEFORE
            #    rendering the text_area so the widget reflects it immediately
            if example_choice != _NONE:
                st.session_state["source_text"] = example_choice

            source_text = st.text_area(
                "Input",
                height=180,
                key="source_text",
                placeholder=placeholder,
                label_visibility="collapsed",
            )
            st.caption(f"{len(source_text.strip())} chars")

            # Language mismatch warning
            lang_valid = _lang_ok(source_text, source_lang)
            if source_text.strip() and not lang_valid:
                if source_lang == "Twi":
                    st.warning(
                        "Wrong language: Twi → Chinese mode expects Twi (Latin) text, "
                        "but Chinese characters were detected."
                    )
                else:
                    st.warning(
                        "Wrong language: Chinese → Twi mode expects Chinese text, "
                        "but Latin/Twi text was detected."
                    )

        # ── output panel ──────────────────────────────────────────────────────
        with col2:
            st.markdown(f"**{target_lang} — Translation**")

            if "translation_output" not in st.session_state:
                st.session_state.translation_output = ""

            st.text_area(
                "Output",
                value=st.session_state.translation_output,
                height=180,
                key="_tgt_display",
                disabled=True,
                label_visibility="collapsed",
            )

            out = st.session_state.translation_output
            if out:
                st.caption(f"{len(out)} chars")
                st.download_button(
                    "Download translation",
                    data=out,
                    file_name="translation.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        # ── action buttons ────────────────────────────────────────────────────
        st.markdown("---")
        btn_col1, btn_col2, _ = st.columns([1, 1, 2])

        translate_ready = bool(source_text.strip()) and lang_valid

        with btn_col1:
            translate_clicked = st.button(
                "Translate",
                use_container_width=True,
                type="primary",
                disabled=not translate_ready,
                help=("Fix the language mismatch above before translating."
                      if not translate_ready and source_text.strip() else None),
            )

        with btn_col2:
            def _clear():
                st.session_state["source_text"]        = ""
                st.session_state["translation_output"] = ""

            st.button("Clear", use_container_width=True, on_click=_clear)

        # ── perform translation ───────────────────────────────────────────────
        result = None
        if translate_clicked and translate_ready:
            result = self._do_translate(source_text.strip(), source_lang,
                                        target_lang, model_dir)
        return result

    def _do_translate(self, text, source_lang, target_lang, model_dir):
        with st.spinner("Translating…"):
            t0 = time.perf_counter()
            try:
                output, confidence = translate(
                    text,
                    direction=model_dir,
                    beam_size=self.config["beam_size"],
                    max_length=self.config["max_length"],
                    alpha=self.config["alpha"],
                )
                elapsed = time.perf_counter() - t0
                st.session_state.translation_output = output
                st.session_state.confidence         = confidence

                st.success(f"Done in {elapsed:.2f}s")

                # Confidence bar
                if confidence is not None:
                    st.markdown(_confidence_bar(confidence),
                                unsafe_allow_html=True)

                return {
                    "source":      text,
                    "translation": output,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "time":        elapsed,
                    "beam_size":   self.config["beam_size"],
                    "confidence":  confidence,
                }
            except Exception as exc:
                st.error(f"Translation failed: {exc}")
                return None
