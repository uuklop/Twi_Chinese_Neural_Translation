"""
Twi <-> Chinese Translation GUI
Loads the best checkpoint and supports both translation directions interactively.

Usage:
    python gui.py
    python gui.py --gpu 0          # run on GPU 0
    python gui.py --cpu            # force CPU
"""
import os
import sys
import pickle
import argparse
import threading
import tkinter as tk
from tkinter import ttk, font as tkfont

import numpy as np
import torch

# ── make sure project root is on path ────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import model as net
import utils
import preprocess
from tokenize_chinese import char_tokenize_line

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR       = os.path.join(ROOT, "data", "twi_chi")
VOCAB_FILE     = os.path.join(DATA_DIR, "twi_chi.vocab.pickle")
BEST_CKPT      = os.path.join(ROOT, "results", "twi_chi_model_best.ckpt")
SPM_MODEL_PATH = os.path.join(DATA_DIR, "twi_spm.model")


# ─────────────────────────────────────────────────────────────────────────────
# Translation engine (loaded once, reused for every request)
# ─────────────────────────────────────────────────────────────────────────────
class TranslationEngine:
    def __init__(self, gpu_id: int = -1):
        self.gpu_id = gpu_id
        self.model  = None
        self.id2w   = None
        self.w2id   = None
        self.sp     = None   # SentencePiece model for Twi BPE
        self.device = "cpu"
        self.epoch  = None
        self.best_score = None

    def load(self, status_cb=None):
        """Load vocab, SPM model and Transformer checkpoint."""
        def log(msg):
            if status_cb:
                status_cb(msg)

        log("Loading vocabulary…")
        with open(VOCAB_FILE, "rb") as f:
            self.id2w = pickle.load(f)
        self.w2id = {w: i for i, w in self.id2w.items()}

        log("Loading SentencePiece BPE model…")
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(SPM_MODEL_PATH)

        log("Loading checkpoint…")
        map_loc = f"cuda:{self.gpu_id}" if self.gpu_id >= 0 and torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(BEST_CKPT, map_location=map_loc, weights_only=False)

        self.epoch      = checkpoint.get("epoch", "?")
        self.best_score = checkpoint.get("best_score", "?")

        config = checkpoint["opts"]
        utils.set_device(self.gpu_id)

        log("Building model…")
        self.model = net.Transformer(config)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        if self.gpu_id >= 0 and torch.cuda.is_available():
            self.model.cuda(self.gpu_id)
            self.device = f"cuda:{self.gpu_id}"
        else:
            self.device = "cpu"

        log(f"Ready  |  epoch {self.epoch}  |  best BLEU {self.best_score:.2f}  |  device: {self.device}")

    # ── helpers ───────────────────────────────────────────────────────────────
    def _ids_to_text(self, ids):
        """Convert a list of token IDs to a single string."""
        return " ".join(self.id2w.get(i, "<unk>") for i in ids if i not in
                        (preprocess.Vocab_Pad.PAD, preprocess.Vocab_Pad.EOS,
                         preprocess.Vocab_Pad.BOS))

    def _post_chinese(self, text):
        """Join adjacent Chinese characters (undo char-level spaces)."""
        tokens = text.split()
        joined = []
        for tok in tokens:
            if (joined
                    and tok
                    and utils._is_chinese_char(tok[0])
                    and utils._is_chinese_char(joined[-1][-1])):
                joined[-1] += tok
            else:
                joined.append(tok)
        return "".join(joined) if all(utils._is_chinese_char(c)
                                      for t in joined for c in t) else " ".join(joined)

    # ── public API ────────────────────────────────────────────────────────────
    def translate(self, text: str, direction: str, beam_size: int = 5) -> str:
        """
        direction: "twi2chi" or "chi2twi"
        Returns the translated string.
        """
        text = text.strip()
        if not text:
            return ""

        if direction == "twi2chi":
            # BPE-tokenise Twi, keep Chinese side as-is
            pieces = self.sp.encode(text, out_type=str)
            ids = [self.w2id.get(p, preprocess.Vocab_Pad.UNK) for p in pieces]
        else:
            # char-tokenise Chinese input
            tokenised = char_tokenize_line(text)
            ids = [self.w2id.get(t, preprocess.Vocab_Pad.UNK)
                   for t in tokenised.split()]

        if not ids:
            return "<empty input after tokenisation>"

        src = [np.array(ids, dtype="i")]
        with torch.no_grad():
            hyp_ids = self.model.translate(src, max_length=100,
                                           beam=beam_size, alpha=0.6)

        raw = self._ids_to_text(hyp_ids[0])

        if direction == "twi2chi":
            return self._post_chinese(raw)
        else:
            # SPM-decode BPE Twi pieces
            pieces = raw.split()
            return self.sp.decode(pieces)


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────
class TranslatorApp(tk.Tk):
    # Colour palette
    BG       = "#1e1e2e"
    PANEL    = "#2a2a3e"
    ACCENT   = "#7c3aed"
    ACCENT_H = "#6d28d9"
    TEXT_FG  = "#e2e8f0"
    MUTED    = "#94a3b8"
    GREEN    = "#22c55e"
    BORDER   = "#3f3f5a"

    def __init__(self, engine: TranslationEngine):
        super().__init__()
        self.engine = engine

        self.title("Twi ↔ Chinese Translator")
        self.configure(bg=self.BG)
        self.geometry("900x640")
        self.minsize(700, 500)
        self.resizable(True, True)

        # fonts
        self._mono  = tkfont.Font(family="Courier New", size=12)
        self._sans  = tkfont.Font(family="Helvetica",   size=12)
        self._title = tkfont.Font(family="Helvetica",   size=16, weight="bold")
        self._small = tkfont.Font(family="Helvetica",   size=10)

        self._direction = tk.StringVar(value="twi2chi")
        self._beam      = tk.IntVar(value=5)
        self._busy      = False

        self._build_ui()
        self._load_model_async()

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # ── top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self, bg=self.PANEL, pady=10, padx=16)
        top.pack(fill=tk.X)

        tk.Label(top, text="Twi ↔ Chinese Translator",
                 font=self._title, bg=self.PANEL, fg=self.TEXT_FG).pack(side=tk.LEFT)

        # beam size spinner (right side of top bar)
        beam_frame = tk.Frame(top, bg=self.PANEL)
        beam_frame.pack(side=tk.RIGHT, padx=(0, 8))
        tk.Label(beam_frame, text="Beam:", font=self._small,
                 bg=self.PANEL, fg=self.MUTED).pack(side=tk.LEFT)
        tk.Spinbox(beam_frame, from_=1, to=10, textvariable=self._beam,
                   width=3, font=self._small,
                   bg=self.PANEL, fg=self.TEXT_FG,
                   buttonbackground=self.PANEL,
                   relief=tk.FLAT).pack(side=tk.LEFT, padx=4)

        # ── direction selector ────────────────────────────────────────────────
        dir_frame = tk.Frame(self, bg=self.BG, pady=10)
        dir_frame.pack(fill=tk.X, padx=16)

        for label, val in [("Twi  →  Chinese", "twi2chi"),
                            ("Chinese  →  Twi",  "chi2twi")]:
            rb = tk.Radiobutton(
                dir_frame, text=label, variable=self._direction, value=val,
                font=self._sans, bg=self.BG, fg=self.TEXT_FG,
                selectcolor=self.ACCENT, activebackground=self.BG,
                activeforeground=self.TEXT_FG,
                command=self._on_direction_change,
                indicatoron=0,
                padx=18, pady=6, bd=0, relief=tk.FLAT,
                highlightthickness=1, highlightbackground=self.BORDER,
            )
            rb.pack(side=tk.LEFT, padx=(0, 8))

        self._update_radio_colors()

        # ── text panels ───────────────────────────────────────────────────────
        panes = tk.Frame(self, bg=self.BG)
        panes.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 8))
        panes.columnconfigure(0, weight=1)
        panes.columnconfigure(1, weight=1)
        panes.rowconfigure(1, weight=1)

        # source label + textarea
        self._src_label = tk.Label(panes, text="Twi (source)",
                                   font=self._small, bg=self.BG, fg=self.MUTED,
                                   anchor="w")
        self._src_label.grid(row=0, column=0, sticky="w", pady=(0, 4))

        src_frame = tk.Frame(panes, bg=self.BORDER, bd=1)
        src_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self._src_text = tk.Text(
            src_frame, font=self._mono, bg=self.PANEL, fg=self.TEXT_FG,
            insertbackground=self.TEXT_FG, wrap=tk.WORD,
            relief=tk.FLAT, padx=10, pady=10,
            highlightthickness=0, bd=0,
        )
        src_scroll = tk.Scrollbar(src_frame, command=self._src_text.yview,
                                  bg=self.PANEL, troughcolor=self.PANEL)
        self._src_text.configure(yscrollcommand=src_scroll.set)
        src_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._src_text.pack(fill=tk.BOTH, expand=True)
        # Ctrl+Enter to translate
        self._src_text.bind("<Control-Return>", lambda _: self._on_translate())

        # target label + textarea
        self._tgt_label = tk.Label(panes, text="Chinese (output)",
                                   font=self._small, bg=self.BG, fg=self.MUTED,
                                   anchor="w")
        self._tgt_label.grid(row=0, column=1, sticky="w", pady=(0, 4))

        tgt_frame = tk.Frame(panes, bg=self.BORDER, bd=1)
        tgt_frame.grid(row=1, column=1, sticky="nsew")
        self._tgt_text = tk.Text(
            tgt_frame, font=self._mono, bg=self.PANEL, fg=self.GREEN,
            wrap=tk.WORD, relief=tk.FLAT, padx=10, pady=10,
            highlightthickness=0, bd=0, state=tk.DISABLED,
        )
        tgt_scroll = tk.Scrollbar(tgt_frame, command=self._tgt_text.yview,
                                  bg=self.PANEL, troughcolor=self.PANEL)
        self._tgt_text.configure(yscrollcommand=tgt_scroll.set)
        tgt_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._tgt_text.pack(fill=tk.BOTH, expand=True)

        # ── bottom bar ────────────────────────────────────────────────────────
        bottom = tk.Frame(self, bg=self.PANEL, pady=8, padx=16)
        bottom.pack(fill=tk.X, side=tk.BOTTOM)

        self._translate_btn = tk.Button(
            bottom, text="Translate  (Ctrl+Enter)",
            font=self._sans, bg=self.ACCENT, fg="white",
            activebackground=self.ACCENT_H, activeforeground="white",
            relief=tk.FLAT, padx=20, pady=6, bd=0,
            command=self._on_translate,
        )
        self._translate_btn.pack(side=tk.LEFT)

        self._clear_btn = tk.Button(
            bottom, text="Clear",
            font=self._sans, bg=self.BORDER, fg=self.TEXT_FG,
            activebackground=self.PANEL, activeforeground=self.TEXT_FG,
            relief=tk.FLAT, padx=14, pady=6, bd=0,
            command=self._on_clear,
        )
        self._clear_btn.pack(side=tk.LEFT, padx=8)

        self._status = tk.Label(
            bottom, text="Loading model…",
            font=self._small, bg=self.PANEL, fg=self.MUTED,
        )
        self._status.pack(side=tk.RIGHT)

    # ── direction toggle ──────────────────────────────────────────────────────
    def _on_direction_change(self):
        self._update_radio_colors()
        d = self._direction.get()
        if d == "twi2chi":
            self._src_label.config(text="Twi (source)")
            self._tgt_label.config(text="Chinese (output)")
        else:
            self._src_label.config(text="Chinese (source)")
            self._tgt_label.config(text="Twi (output)")
        self._on_clear()

    def _update_radio_colors(self):
        """Highlight the active radio button."""
        for widget in self.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Radiobutton):
                        selected = child["value"] == self._direction.get()
                        child.config(
                            bg=self.ACCENT if selected else self.BG,
                            fg="white"     if selected else self.MUTED,
                        )

    # ── clear ────────────────────────────────────────────────────────────────
    def _on_clear(self):
        self._src_text.delete("1.0", tk.END)
        self._set_output("")

    # ── status bar update (thread-safe) ───────────────────────────────────────
    def _set_status(self, msg: str, color: str = None):
        self.after(0, lambda: self._status.config(
            text=msg, fg=color or self.MUTED))

    def _set_output(self, text: str):
        self._tgt_text.config(state=tk.NORMAL)
        self._tgt_text.delete("1.0", tk.END)
        if text:
            self._tgt_text.insert(tk.END, text)
        self._tgt_text.config(state=tk.DISABLED)

    # ── model loading ─────────────────────────────────────────────────────────
    def _load_model_async(self):
        self._translate_btn.config(state=tk.DISABLED)

        def _worker():
            try:
                self.engine.load(status_cb=lambda m: self._set_status(m))
                self.after(0, lambda: self._translate_btn.config(state=tk.NORMAL))
                self._set_status(
                    f"Ready  |  epoch {self.engine.epoch}  |  "
                    f"best BLEU {self.engine.best_score:.2f}  |  {self.engine.device}",
                    color=self.GREEN,
                )
            except Exception as exc:
                self._set_status(f"Load error: {exc}", color="#ef4444")

        threading.Thread(target=_worker, daemon=True).start()

    # ── translation ───────────────────────────────────────────────────────────
    def _on_translate(self):
        if self._busy or self.engine.model is None:
            return

        src = self._src_text.get("1.0", tk.END).strip()
        if not src:
            return

        self._busy = True
        self._translate_btn.config(state=tk.DISABLED, text="Translating…")
        self._set_status("Translating…")
        self._set_output("")

        direction = self._direction.get()
        beam      = self._beam.get()

        def _worker():
            try:
                result = self.engine.translate(src, direction=direction,
                                               beam_size=beam)
                self.after(0, lambda: self._set_output(result))
                self._set_status("Done.", color=self.GREEN)
            except Exception as exc:
                self.after(0, lambda: self._set_output(f"[Error] {exc}"))
                self._set_status(f"Error: {exc}", color="#ef4444")
            finally:
                self._busy = False
                self.after(0, lambda: self._translate_btn.config(
                    state=tk.NORMAL, text="Translate  (Ctrl+Enter)"))

        threading.Thread(target=_worker, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=-1,
                   help="GPU id (default: -1 = CPU)")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even if GPU is available")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gpu_id = -1 if args.cpu else args.gpu

    engine = TranslationEngine(gpu_id=gpu_id)
    app    = TranslatorApp(engine)
    app.mainloop()
