"""Sidebar component — configuration and model status."""

import streamlit as st


class Sidebar:
    """Renders the sidebar with configuration options."""

    def render(self):
        with st.sidebar:
            st.markdown("## Configuration")

            st.markdown("#### Direction")
            direction = st.radio(
                "Select direction:",
                ["Twi → Chinese", "Chinese → Twi"],
                index=0,
                label_visibility="collapsed",
            )

            st.markdown("#### Model Settings")

            beam_size = st.slider(
                "Beam Size",
                min_value=1, max_value=10, value=5,
                help="Higher values may improve quality but slow down translation",
            )

            max_length = st.slider(
                "Max Output Length",
                min_value=10, max_value=200, value=100,
                help="Maximum number of tokens in the generated translation",
            )

            alpha = st.slider(
                "Length Penalty (α)",
                min_value=0.1, max_value=2.0, value=0.6, step=0.1,
                help="Lower values favour shorter output; higher values favour longer",
            )

            st.markdown("---")
            if st.button("Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

            st.markdown("---")
            st.markdown("#### Model Status")
            from app_utils.model_loader import engine_info
            info = engine_info()
            if info:
                bleu = info["best_score"] * 100
                st.success(
                    f"Model ready  ·  Translation quality: {bleu:.1f} BLEU"
                )
            else:
                st.warning("Model not yet loaded")

            st.markdown("---")
            st.markdown("#### About")
            st.markdown(
                """
Bidirectional Twi ↔ Chinese neural machine translation.

**Architecture**
- Transformer (6 layers, 8 heads, d=512)
- SentencePiece BPE — Twi tokenization
- Character-level — Chinese tokenization
- Direction tags: `<2zh>` / `<2tw>`
- Shared bidirectional vocabulary (~9 K tokens)
                """
            )

        return {
            "direction":  direction,
            "beam_size":  beam_size,
            "max_length": max_length,
            "alpha":      alpha,
        }
