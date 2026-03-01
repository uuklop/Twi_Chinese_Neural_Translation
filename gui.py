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
import time
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
# Translation engine
# ─────────────────────────────────────────────────────────────────────────────
class TranslationEngine:
    def __init__(self, gpu_id: int = -1):
        self.gpu_id     = gpu_id
        self.model      = None
        self.id2w       = None
        self.w2id       = None
        self.sp         = None
        self.device     = "cpu"
        self.epoch      = None
        self.best_score = None   # stored as fraction [0,1]; display as ×100

    def load(self, status_cb=None):
        def log(msg):
            if status_cb:
                status_cb(msg)

        # ── vocab ──────────────────────────────────────────────────────────
        if not os.path.exists(VOCAB_FILE):
            raise FileNotFoundError(
                f"Vocab file not found: {VOCAB_FILE}\n"
                "Run  bash pipeline.sh preprocess  first."
            )
        log("Loading vocabulary…")
        with open(VOCAB_FILE, "rb") as f:
            self.id2w = pickle.load(f)
        self.w2id = {w: i for i, w in self.id2w.items()}

        # ── SentencePiece ──────────────────────────────────────────────────
        if not os.path.exists(SPM_MODEL_PATH):
            raise FileNotFoundError(
                f"SPM model not found: {SPM_MODEL_PATH}\n"
                "Run  bash pipeline.sh prepare  first."
            )
        log("Loading SentencePiece model…")
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(SPM_MODEL_PATH)

        # ── checkpoint ────────────────────────────────────────────────────
        if not os.path.exists(BEST_CKPT):
            raise FileNotFoundError(
                f"No checkpoint found at: {BEST_CKPT}\n"
                "Training may still be in progress. "
                "Run  bash pipeline.sh train  and wait for the first evaluation."
            )
        log("Loading checkpoint…")
        map_loc = (f"cuda:{self.gpu_id}"
                   if self.gpu_id >= 0 and torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(BEST_CKPT, map_location=map_loc,
                                weights_only=False)

        self.epoch      = checkpoint.get("epoch", "?")
        raw_score       = checkpoint.get("best_score", 0.0)
        self.best_score = float(raw_score)   # ensure plain Python float

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

    # ── helpers ───────────────────────────────────────────────────────────────
    def _ids_to_text(self, ids):
        return " ".join(
            self.id2w.get(i, "<unk>") for i in ids
            if i not in (preprocess.Vocab_Pad.PAD,
                         preprocess.Vocab_Pad.EOS,
                         preprocess.Vocab_Pad.BOS)
        )

    def _post_chinese(self, text):
        tokens = text.split()
        joined = []
        for tok in tokens:
            if (joined and tok
                    and utils._is_chinese_char(tok[0])
                    and utils._is_chinese_char(joined[-1][-1])):
                joined[-1] += tok
            else:
                joined.append(tok)
        return ("".join(joined)
                if all(utils._is_chinese_char(c)
                       for t in joined for c in t)
                else " ".join(joined))

    # ── public API ────────────────────────────────────────────────────────────
    def translate(self, text: str, direction: str, beam_size: int = 5) -> str:
        text = text.strip()
        if not text:
            return ""

        if direction == "twi2chi":
            pieces = self.sp.encode(text, out_type=str)
            if not pieces:
                return "<empty input after tokenisation>"
            # <2zh> tells the model to generate Chinese output
            tag_id = self.w2id.get('<2zh>', preprocess.Vocab_Pad.UNK)
            ids = [tag_id] + [self.w2id.get(p, preprocess.Vocab_Pad.UNK)
                              for p in pieces]
        else:
            tokenised = char_tokenize_line(text)
            tokens = tokenised.split()
            if not tokens:
                return "<empty input after tokenisation>"
            # <2tw> tells the model to generate Twi output
            tag_id = self.w2id.get('<2tw>', preprocess.Vocab_Pad.UNK)
            ids = [tag_id] + [self.w2id.get(t, preprocess.Vocab_Pad.UNK)
                              for t in tokens]

        src = [np.array(ids, dtype="i")]
        with torch.no_grad():
            hyp_ids = self.model.translate(src, max_length=100,
                                           beam=beam_size, alpha=0.6)

        raw = self._ids_to_text(hyp_ids[0])
        if direction == "twi2chi":
            return self._post_chinese(raw)
        else:
            return self.sp.decode(raw.split())


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────
class TranslatorApp(tk.Tk):
    BG       = "#1e1e2e"
    PANEL    = "#2a2a3e"
    ACCENT   = "#7c3aed"
    ACCENT_H = "#6d28d9"
    TEXT_FG  = "#e2e8f0"
    MUTED    = "#94a3b8"
    GREEN    = "#22c55e"
    RED      = "#ef4444"
    BORDER   = "#3f3f5a"

    def __init__(self, engine: TranslationEngine):
        super().__init__()
        self.engine = engine

        self.title("Twi ↔ Chinese Translator")
        self.configure(bg=self.BG)
        self.geometry("940x660")
        self.minsize(700, 500)
        self.resizable(True, True)

        self._mono  = tkfont.Font(family="Courier New", size=12)
        self._sans  = tkfont.Font(family="Helvetica",   size=12)
        self._title = tkfont.Font(family="Helvetica",   size=16, weight="bold")
        self._small = tkfont.Font(family="Helvetica",   size=10)

        self._direction  = tk.StringVar(value="twi2chi")
        self._beam       = tk.IntVar(value=5)
        self._busy       = False
        self._radio_btns = []   # keep refs for reliable color updates

        self._build_ui()
        self._load_model_async()

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # ── top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self, bg=self.PANEL, pady=10, padx=16)
        top.pack(fill=tk.X)

        tk.Label(top, text="Twi ↔ Chinese Translator",
                 font=self._title, bg=self.PANEL, fg=self.TEXT_FG).pack(side=tk.LEFT)

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
                indicatoron=0, padx=18, pady=6, bd=0, relief=tk.FLAT,
                highlightthickness=1, highlightbackground=self.BORDER,
            )
            rb.pack(side=tk.LEFT, padx=(0, 8))
            self._radio_btns.append((rb, val))

        self._update_radio_colors()

        # ── text panels ───────────────────────────────────────────────────────
        panes = tk.Frame(self, bg=self.BG)
        panes.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 4))
        panes.columnconfigure(0, weight=1)
        panes.columnconfigure(1, weight=1)
        panes.rowconfigure(1, weight=1)

        # source
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
        self._src_text.bind("<Control-Return>", lambda _: self._on_translate())
        self._src_text.bind("<KeyRelease>",     lambda _: self._update_char_count())

        # target
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

        # ── char count bar ────────────────────────────────────────────────────
        count_bar = tk.Frame(self, bg=self.BG)
        count_bar.pack(fill=tk.X, padx=16)
        self._char_count = tk.Label(count_bar, text="",
                                    font=self._small, bg=self.BG, fg=self.MUTED,
                                    anchor="w")
        self._char_count.pack(side=tk.LEFT)

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

        self._copy_btn = tk.Button(
            bottom, text="Copy Output",
            font=self._sans, bg=self.BORDER, fg=self.TEXT_FG,
            activebackground=self.PANEL, activeforeground=self.TEXT_FG,
            relief=tk.FLAT, padx=14, pady=6, bd=0,
            command=self._on_copy,
        )
        self._copy_btn.pack(side=tk.LEFT)

        self._status = tk.Label(
            bottom, text="Loading model…",
            font=self._small, bg=self.PANEL, fg=self.MUTED,
        )
        self._status.pack(side=tk.RIGHT)

    # ── direction toggle ──────────────────────────────────────────────────────
    def _on_direction_change(self):
        self._update_radio_colors()
        if self._direction.get() == "twi2chi":
            self._src_label.config(text="Twi (source)")
            self._tgt_label.config(text="Chinese (output)")
        else:
            self._src_label.config(text="Chinese (source)")
            self._tgt_label.config(text="Twi (output)")
        self._on_clear()

    def _update_radio_colors(self):
        active = self._direction.get()
        for rb, val in self._radio_btns:
            selected = (val == active)
            rb.config(bg=self.ACCENT if selected else self.BG,
                      fg="white"     if selected else self.MUTED)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _on_clear(self):
        self._src_text.delete("1.0", tk.END)
        self._set_output("")
        self._char_count.config(text="")

    def _on_copy(self):
        text = self._tgt_text.get("1.0", tk.END).strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self._set_status("Copied to clipboard.", color=self.GREEN)

    def _update_char_count(self):
        n = len(self._src_text.get("1.0", tk.END).strip())
        self._char_count.config(text=f"{n} chars" if n else "")

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
                score_str = (f"{self.engine.best_score * 100:.2f}"
                             if self.engine.best_score is not None else "n/a")
                self.after(0, lambda: self._translate_btn.config(state=tk.NORMAL))
                self._set_status(
                    f"Ready  |  epoch {self.engine.epoch}  |  "
                    f"best BLEU {score_str}  |  {self.engine.device}",
                    color=self.GREEN,
                )
            except FileNotFoundError as exc:
                self._set_status(str(exc).split("\n")[0], color=self.RED)
                self.after(0, lambda: self._set_output(
                    f"Model not ready:\n\n{exc}"))
            except Exception as exc:
                self._set_status(f"Load error: {exc}", color=self.RED)

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
        t0        = time.perf_counter()

        def _worker():
            try:
                result   = self.engine.translate(src, direction=direction,
                                                 beam_size=beam)
                elapsed  = time.perf_counter() - t0
                self.after(0, lambda: self._set_output(result))
                self._set_status(f"Done  ({elapsed:.2f}s)", color=self.GREEN)
            except Exception as exc:
                self.after(0, lambda: self._set_output(f"[Error] {exc}"))
                self._set_status(f"Error: {exc}", color=self.RED)
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
    args   = parse_args()
    gpu_id = -1 if args.cpu else args.gpu

    engine = TranslationEngine(gpu_id=gpu_id)
    app    = TranslatorApp(engine)
    app.mainloop()
