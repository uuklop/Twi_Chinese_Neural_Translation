"""Model loading and inference — mirrors gui.py TranslationEngine exactly."""

import os
import sys
import pickle
import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*Examining the path.*')

import numpy as np
import torch
import streamlit as st
from pathlib import Path

# ── nmt_core on sys.path (self-contained, no parent-dir dependency) ───────────
APP_DIR = Path(__file__).parent.parent          # streamlit_app/
NMT_DIR = APP_DIR / "nmt_core"
sys.path.insert(0, str(NMT_DIR))

import preprocess
from tokenize_chinese import char_tokenize_line

# ── model artifact paths (override with env vars for different deployments) ───
#   MODEL_DIR  → directory containing all three files (default: app/models/)
#   MODEL_CKPT → checkpoint filename inside MODEL_DIR
#   MODEL_VOCAB → vocab pickle filename inside MODEL_DIR
#   MODEL_SPM  → SentencePiece model filename inside MODEL_DIR
_MODEL_DIR = Path(os.getenv("MODEL_DIR", str(APP_DIR / "models")))
CKPT_PATH  = _MODEL_DIR / os.getenv("MODEL_CKPT",  "twi_chi_model_best.ckpt")
VOCAB_PATH = _MODEL_DIR / os.getenv("MODEL_VOCAB", "twi_chi.vocab.pickle")
SPM_PATH   = _MODEL_DIR / os.getenv("MODEL_SPM",   "twi_spm.model")


# ── HF Hub auto-download ──────────────────────────────────────────────────────
def _hf_secret(key: str):
    """Read a value from Streamlit secrets, return None if unavailable."""
    try:
        return st.secrets.get(key)
    except Exception:
        return None


def _ensure_model_files():
    """Download any missing model files from Hugging Face Hub.

    Requires HF_REPO_ID to be set as an environment variable or
    Streamlit secret (e.g.  username/twi-chi-model).
    Optionally set HF_TOKEN for private repos.
    """
    needed = [
        (CKPT_PATH,  os.getenv("MODEL_CKPT",  "twi_chi_model_best.ckpt")),
        (VOCAB_PATH, os.getenv("MODEL_VOCAB", "twi_chi.vocab.pickle")),
        (SPM_PATH,   os.getenv("MODEL_SPM",   "twi_spm.model")),
    ]
    missing = [(path, fname) for path, fname in needed if not path.exists()]
    if not missing:
        return

    hf_repo = os.getenv("HF_REPO_ID") or _hf_secret("HF_REPO_ID")
    if not hf_repo:
        missing_names = ", ".join(fname for _, fname in missing)
        raise FileNotFoundError(
            f"Model file(s) not found: {missing_names}\n\n"
            "To auto-download, set  HF_REPO_ID  to your Hugging Face repo\n"
            "(e.g. 'username/twi-chi-model') as an environment variable or\n"
            "Streamlit secret, then reboot the app.\n\n"
            "Alternatively, place the three model files in  models/"
        )

    from huggingface_hub import hf_hub_download
    hf_token = os.getenv("HF_TOKEN") or _hf_secret("HF_TOKEN")
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for local_path, filename in missing:
        with st.spinner(f"Downloading {filename} from Hugging Face Hub…"):
            hf_hub_download(
                repo_id=hf_repo,
                filename=filename,
                local_dir=str(_MODEL_DIR),
                token=hf_token,
            )


# ── cached engine load (once per Streamlit server lifetime) ───────────────────
@st.cache_resource(show_spinner="Loading translation model…")
def load_engine():
    """Load the single bidirectional model and return all inference assets."""
    import model as net

    _ensure_model_files()

    for path, label in [(CKPT_PATH,  "Checkpoint"),
                        (VOCAB_PATH, "Vocabulary"),
                        (SPM_PATH,   "SentencePiece model")]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found: {path}\n"
                "Place the model files in the  models/  directory, or set\n"
                "MODEL_DIR / MODEL_CKPT / MODEL_VOCAB / MODEL_SPM env vars."
            )

    # vocab: pickle stores id2w  {int → str}
    with open(VOCAB_PATH, "rb") as f:
        id2w = pickle.load(f)
    w2id = {w: i for i, w in id2w.items()}

    # SentencePiece
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(SPM_PATH))

    # checkpoint
    checkpoint = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    config = checkpoint["opts"]

    m = net.Transformer(config)
    m.load_state_dict(checkpoint["state_dict"])
    m.eval()

    return {
        "model":      m,
        "id2w":       id2w,
        "w2id":       w2id,
        "sp":         sp,
        "epoch":      checkpoint.get("epoch", "?"),
        "best_score": float(checkpoint.get("best_score", 0.0)),
    }


# ── helpers ───────────────────────────────────────────────────────────────────
def _ids_to_text(ids, id2w):
    PAD = preprocess.Vocab_Pad.PAD
    EOS = preprocess.Vocab_Pad.EOS
    BOS = preprocess.Vocab_Pad.BOS
    return " ".join(id2w.get(i, "<unk>") for i in ids if i not in (PAD, EOS, BOS))


def _post_chinese(text):
    """Join adjacent Chinese character tokens into a single string."""
    from utils import _is_chinese_char
    tokens = text.split()
    joined = []
    for tok in tokens:
        if (joined and tok
                and _is_chinese_char(tok[0])
                and _is_chinese_char(joined[-1][-1])):
            joined[-1] += tok
        else:
            joined.append(tok)
    return ("".join(joined)
            if all(_is_chinese_char(c) for t in joined for c in t)
            else " ".join(joined))


# ── confidence score ──────────────────────────────────────────────────────────
def _score_to_confidence(score) -> float:
    """Map a normalized beam log-prob score to a [0, 1] confidence value.

    Beam scores are length-normalised log probabilities (≤ 0).
    We use  exp(score / 2)  so that:
      score =  0  → 100 %   (perfect)
      score = -1  → ~61 %   (good)
      score = -2  → ~37 %   (moderate)
      score = -4  → ~14 %   (poor)
    """
    import math
    try:
        s = float(score)
        return max(0.0, min(1.0, math.exp(s / 2.0)))
    except Exception:
        return 0.0


# ── public translation function ───────────────────────────────────────────────
def translate(text: str, direction: str,
              beam_size: int = 5, max_length: int = 100,
              alpha: float = 0.6) -> tuple:
    """Translate text.  Returns (output_text, confidence_float).

    direction: 'twi2chi' or 'chi2twi'
    confidence is in [0, 1]; None if unavailable.
    """
    text = text.strip()
    if not text:
        return "", None

    engine = load_engine()
    w2id = engine["w2id"]
    id2w = engine["id2w"]
    sp   = engine["sp"]
    m    = engine["model"]

    UNK = preprocess.Vocab_Pad.UNK

    if direction == "twi2chi":
        pieces = sp.encode(text, out_type=str)
        if not pieces:
            return "<empty input after tokenisation>", None
        tag_id = w2id.get("<2zh>", UNK)
        ids = [tag_id] + [w2id.get(p, UNK) for p in pieces]
    else:
        tokenised = char_tokenize_line(text)
        tokens = tokenised.split()
        if not tokens:
            return "<empty input after tokenisation>", None
        tag_id = w2id.get("<2tw>", UNK)
        ids = [tag_id] + [w2id.get(t, UNK) for t in tokens]

    src = [np.array(ids, dtype="i")]
    with torch.no_grad():
        hyp_ids, scores = m.translate(src, max_length=max_length,
                                      beam=beam_size, alpha=alpha)

    raw = _ids_to_text(hyp_ids[0], id2w)
    if direction == "twi2chi":
        output = _post_chinese(raw)
    else:
        output = sp.decode(raw.split())

    confidence = _score_to_confidence(scores[0]) if scores else None
    return output, confidence


def engine_info() -> dict:
    """Return model metadata (epoch, best BLEU). Returns {} if not loaded."""
    try:
        e = load_engine()
        return {"epoch": e["epoch"], "best_score": e["best_score"]}
    except Exception:
        return {}
