# Model Files

The three trained model artifacts are required to run the app.

| File | Size | Description |
|------|------|-------------|
| `twi_chi_model_best.ckpt` | ~578 MB | Trained Transformer checkpoint (best validation BLEU) |
| `twi_chi.vocab.pickle` | ~96 KB | Shared vocabulary `{int → token}` |
| `twi_spm.model` | ~297 KB | SentencePiece BPE model for Twi tokenisation |

These files are **gitignored** — never commit large binaries to git.

---

## Option A — Streamlit Cloud (recommended): Hugging Face Hub

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Create a new model repository (can be private)
3. Upload the three files to the repo root
4. In Streamlit Cloud → **App settings → Secrets**, add:

```toml
HF_REPO_ID = "JamesSalar/twi-chinese"
# Only needed for private repos:
# HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
```

The app will auto-download the files on first boot and cache them.

---

## Option B — Local development

Place the files directly in this `models/` directory:

```bash
cp results/twi_chi_model_best.ckpt   streamlit_app/models/
cp data/twi_chi/twi_chi.vocab.pickle streamlit_app/models/
cp data/twi_chi/twi_spm.model        streamlit_app/models/
```

Then run:

```bash
cd streamlit_app
streamlit run app.py
```

---

## Option C — Custom path via environment variables

```bash
export MODEL_DIR=/path/to/your/models
streamlit run app.py
```
