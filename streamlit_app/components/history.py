"""Translation history component."""

import streamlit as st


class TranslationHistory:
    """Renders the translation history panel."""

    def __init__(self, history: list):
        self.history = history

    def render(self):
        if not self.history:
            return

        st.markdown("---")
        st.markdown("### Recent Translations")

        for i, item in enumerate(self.history[:10]):
            label = (
                f"{item['source_lang']} → {item['target_lang']}  "
                f"·  {item['time']:.2f}s  ·  beam {item['beam_size']}"
            )
            with st.expander(label, expanded=(i == 0)):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**{item['source_lang']}**")
                    st.code(item["source"][:500], language=None)
                    st.download_button(
                        "Download source",
                        data=item["source"],
                        file_name=f"source_{i}.txt",
                        mime="text/plain",
                        key=f"dl_src_{i}",
                    )

                with col2:
                    st.markdown(f"**{item['target_lang']}**")
                    st.code(item["translation"][:500], language=None)
                    st.download_button(
                        "Download translation",
                        data=item["translation"],
                        file_name=f"translation_{i}.txt",
                        mime="text/plain",
                        key=f"dl_tgt_{i}",
                    )
