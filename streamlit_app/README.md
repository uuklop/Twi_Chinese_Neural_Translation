# Twi ↔ Chinese Translator — Web App

Streamlit-based web interface for the Twi ↔ Chinese neural machine translation model.
This directory is **self-contained**: it does not depend on the training code or any parent directory.

---

## Quick Start (Local)

```bash
cd streamlit_app
pip install -r requirements.txt

# Place the three model files in streamlit_app/models/
#   twi_chi_model_best.ckpt   (~578 MB — trained checkpoint)
#   twi_chi.vocab.pickle       (~96 KB  — shared vocabulary)
#   twi_spm.model              (~297 KB — SentencePiece BPE model for Twi)

streamlit run app.py
# open http://localhost:8501
```

---

## Docker

```bash
docker-compose up --build -d
# open http://localhost:8501
```

The compose file mounts `./models` read-only into the container.
Place the three model files in `streamlit_app/models/` before running.

---

## Streamlit Cloud Deployment

1. Fork the parent repository.
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud).
3. Set the **main file path** to `streamlit_app/app.py`.
4. Add a `MODEL_DIR` secret/env var pointing to your model files,
   or upload them to `streamlit_app/models/` before deploying.

---

## Model Files

The three required model files are **not tracked by git** (see `.gitignore`).
See `models/README.md` for instructions on obtaining or placing them.

| File | Size | Description |
|---|---|---|
| `twi_chi_model_best.ckpt` | ~578 MB | Best averaged checkpoint |
| `twi_chi.vocab.pickle` | ~96 KB | Shared vocabulary (source + target) |
| `twi_spm.model` | ~297 KB | SentencePiece model for Twi tokenisation |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | `models/` (relative to app) | Directory containing model files |
| `MODEL_CKPT` | `twi_chi_model_best.ckpt` | Checkpoint filename |
| `MODEL_VOCAB` | `twi_chi.vocab.pickle` | Vocabulary filename |
| `MODEL_SPM` | `twi_spm.model` | SentencePiece model filename |

---

## Structure

```
streamlit_app/
├── app.py                  # Entry point
├── app_utils/
│   └── model_loader.py     # Model loading & inference (env-var configurable)
├── components/
│   ├── sidebar.py          # Config sidebar (direction, beam, max-len, alpha)
│   ├── translator.py       # Translation UI + confidence bar + example picker
│   └── history.py          # Session history
├── nmt_core/               # Self-contained NMT inference code
│   ├── model.py            # Transformer (returns score alongside output)
│   ├── decoding.py         # Beam search & greedy decoding
│   └── ...
├── assets/style.css        # Custom CSS
├── data/
│   ├── examples_twi.txt    # 500 example Twi sentences
│   └── examples_chi.txt    # 500 example Chinese sentences
├── images/                 # Flag images (Ghana, China)
├── models/                 # Place trained model files here (gitignored)
│   ├── .gitkeep
│   └── README.md
├── .streamlit/config.toml  # Streamlit server config
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run.sh                  # Local run helper
└── TROUBLESHOOTING.md
```

---

## Features

- **Bidirectional**: Twi → Chinese and Chinese → Twi
- **Language validation**: warns if wrong script is entered
- **Confidence bar**: colour-coded score (High / Medium / Low) after each translation
- **500 examples** per language in a scrollable dropdown (everyday + biblical sentences)
- **Session history**: expandable log of past translations with download
- **Configurable**: beam size, max output length, length penalty (α)
- **Download**: export any translation as a `.txt` file

---

## Remote Access

The server binds to `0.0.0.0` by default. To access from another machine:

```
http://<SERVER_IP>:8501
```

Open port 8501 on your firewall if needed:

```bash
sudo ufw allow 8501/tcp
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more details.

---

© trudey@uestc2025
