# Twi ↔ Chinese Neural Machine Translation

A bidirectional neural machine translation system between **Twi** (Akan, Ghana) and **Mandarin Chinese** — two languages underrepresented in NMT research. Built on a modernised Transformer architecture trained as a single shared model for both directions simultaneously.

---

## Architecture

Based on *[Attention Is All You Need](https://arxiv.org/abs/1706.03762)* (Vaswani et al., 2017), with the following modernisations:

| Component | Original | This Work |
|---|---|---|
| Attention | Scaled dot-product | **Flash Attention** (`F.scaled_dot_product_attention`) |
| Position encoding | Sinusoidal (absolute) | **Rotary Position Embedding (RoPE)** |
| FFN activation | ReLU | **SwiGLU** |
| Training precision | FP32 | **Automatic Mixed Precision (AMP, FP16)** |
| Model selection | Single best checkpoint | **Checkpoint averaging** (last 8 checkpoints) |
| Gradient clipping | None | `clip_grad_norm_` (max = 1.0) |

**Configuration** (default): 6 layers · 8 attention heads · d_model = 512 · d_ffn = 2048 · dropout = 0.1 · label smoothing = 0.1

---

## Repository Structure

```
.
├── data/
│   ├── twi_chi/                   # Processed dataset
│   │   ├── val.src / val.tgt      # Validation set — Twi→Chi (500 pairs)
│   │   ├── test.src / test.tgt    # Test set — Twi→Chi (500 pairs)
│   │   ├── test_rev.src / .tgt    # Test set — Chi→Twi (reversed)
│   │   ├── twi_spm.model          # SentencePiece BPE model for Twi
│   │   └── twi_spm.vocab          # SentencePiece vocabulary
│   └── more_raw_twi_chinese_pairs/
│       └── twi_chinese_direct.csv # Source parallel data (2.1 MB)
│
├── img/                           # Architecture diagrams
│
├── # ── Model ──────────────────────────────────────────────
├── model.py                         # Transformer model (encoder, decoder, attention)
├── decoding.py             # Beam search & greedy decoding
├── pad_utils.py                # PadRemover utility for efficient FFN
│
├── # ── Training pipeline ──────────────────────────────────
├── config.py                      # Argument parsing (train / translate / preprocess)
├── optimizer.py                   # Warmup Adam with AMP-aware grad clipping
├── metrics.py                   # BLEU, WER, CER evaluation
├── train.py                       # Training loop with TensorBoard logging
├── translate.py                   # Batch translation script
├── preprocess.py                  # Tokenise & serialise dataset to .npy
├── utils.py                       # Batching, padding, statistics, post-processing
│
├── # ── Data preparation ───────────────────────────────────
├── tokenize_chinese.py                # Chinese character-level tokeniser
├── build_dataset.py               # Build bidirectional train/val/test splits
├── build_bpe.py                 # Train SentencePiece BPE on Twi; apply to data
│
├── # ── Utilities ──────────────────────────────────────────
├── gui.py                         # Tkinter desktop GUI for interactive translation
├── plot_training.py                # Plot training curves from results/metrics.jsonl
├── pipeline.sh                 # End-to-end pipeline script
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone <repo-url>
cd <repo>
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+ (CUDA recommended), SentencePiece, Matplotlib, TensorBoard.

---

## Quick Start

Run the entire pipeline (prepare → preprocess → train → translate) with one command:

```bash
bash pipeline.sh          # all stages
bash pipeline.sh prepare  # Step 1 only: build splits + BPE
bash pipeline.sh preprocess  # Step 2 only
bash pipeline.sh train    # Step 3 only
bash pipeline.sh translate   # Step 4 only
```

Set `GPU=0` at the top of `pipeline.sh` for GPU training (default) or `GPU=-1` for CPU.

---

## Step-by-Step Usage

### Step 1a — Build bidirectional data splits

Pools existing pairs with the source CSV, deduplicates, and creates a bidirectional training set (Twi→Chi + Chi→Twi shuffled together), plus held-out val/test sets (500 pairs each, Twi→Chi only).

```bash
python build_dataset.py
```

Output files in `data/twi_chi/`: `train.src`, `train.tgt`, `val.src`, `val.tgt`, `test.src`, `test.tgt`, `test_rev.src`, `test_rev.tgt`.

### Step 1b — Train SentencePiece BPE and apply to Twi

Trains a BPE model (vocab size 4000) on the Twi side of the training data, then applies it in-place. Chinese lines are left as character-level tokens.

```bash
python build_bpe.py
```

### Step 2 — Preprocess (build integer-indexed .npy files)

```bash
python preprocess.py \
    -i data/twi_chi \
    -s-train train.src  -t-train train.tgt \
    -s-valid val.src    -t-valid val.tgt \
    -s-test  test.src   -t-test  test.tgt \
    --save_data twi_chi
```

### Step 3 — Train

```bash
python train.py \
    -i data/twi_chi \
    --data twi_chi \
    --wbatchsize 4096 \
    --batchsize 60 \
    --tied \
    --beam_size 5 \
    --epoch 60 \
    --layers 6 \
    --multi_heads 8 \
    --warmup_steps 4000 \
    --gpu 0 \
    --out results \
    --model_file      results/twi_chi_model.ckpt \
    --best_model_file results/twi_chi_model_best.ckpt \
    --dev_hyp         results/twi_chi_valid.out \
    --dev_hyp_rev     results/twi_chi_valid_rev.out \
    --test_hyp        results/twi_chi_test.out \
    --test_hyp_rev    results/twi_chi_test_rev.out \
    --spm_model       data/twi_chi/twi_spm.model
```

### Step 4 — Translate

```bash
# Twi → Chinese
python translate.py \
    -i data/twi_chi --data twi_chi \
    --batchsize 60 --beam_size 5 \
    --best_model_file results/twi_chi_model_best.ckpt \
    --model_file      results/twi_chi_model.ckpt \
    --src  data/twi_chi/test.src \
    --output results/twi_chi_pred_twi2chi.txt \
    --gpu 0

# Chinese → Twi
python translate.py \
    -i data/twi_chi --data twi_chi \
    --batchsize 60 --beam_size 5 \
    --best_model_file results/twi_chi_model_best.ckpt \
    --model_file      results/twi_chi_model.ckpt \
    --src  data/twi_chi/test_rev.src \
    --output results/twi_chi_pred_chi2twi.txt \
    --spm_model data/twi_chi/twi_spm.model \
    --gpu 0
```

---

## Visualisation

### TensorBoard

TensorBoard logs are written to `results/runs/` automatically during training.

```bash
tensorboard --logdir results/runs
# open http://localhost:6006
```

**Logged signals:**

| Tab | Metric | Frequency |
|---|---|---|
| Scalars | `Step/loss`, `Step/ppl`, `Step/grad_norm` | Every training step |
| Scalars | `Optimizer/learning_rate` | Every training step |
| Scalars | `Perplexity/{train,val}`, `Accuracy/{train,val}` | Every eval checkpoint |
| Scalars | `BLEU/{twi_to_chinese,chinese_to_twi,average}` | Every eval checkpoint |
| Histograms | Weight & gradient distributions per layer | Every eval checkpoint |
| Text | Live sample translations (5 pairs, both directions) | Every eval checkpoint |
| HParams | All hyperparameters | Once at startup |

### Static plots

```bash
python plot_training.py                          # interactive window
python plot_training.py --save results/metrics.png   # save to file
```

Produces a 2×2 dashboard: perplexity · accuracy · BLEU · learning rate.

---

## Interactive GUI

Launch a desktop translation interface backed by the best checkpoint:

```bash
python gui.py           # auto-detect GPU
python gui.py --gpu 0   # explicit GPU
python gui.py --cpu     # force CPU
```

Supports both directions, adjustable beam size, and Ctrl+Enter to translate.

---

## Data

| Split | Pairs | Notes |
|---|---|---|
| Train | ~83 000 | Bidirectional (Twi→Chi + Chi→Twi shuffled) |
| Validation | 500 | Twi→Chi only |
| Test | 500 | Twi→Chi only; `test_rev.*` for Chi→Twi |

**Tokenisation:**
- **Twi**: SentencePiece BPE, vocab size 4 000, `character_coverage=1.0`
- **Chinese**: Character-level (each character = one token, no segmentation ambiguity)

**Data leakage:** Val and test sets are strictly disjoint from the training set (enforced in `build_dataset.py`).

**Source data:** `data/more_raw_twi_chinese_pairs/twi_chinese_direct.csv` (included).
The 99 MB supplementary CSV (`academic_reverse_pivot.csv`) is excluded from git due to size.

---

## Reproducing from scratch

If you only have the source CSV and want to rebuild everything:

```bash
bash pipeline.sh prepare     # splits + BPE
bash pipeline.sh preprocess  # .npy files
bash pipeline.sh train       # model
bash pipeline.sh translate   # output files
```

---

## Acknowledgements

- Original Transformer implementation adapted from [DevSinghSachan/multilingual_nmt](https://github.com/DevSinghSachan/multilingual_nmt), which in turn draws from [Sosuke Kobayashi's Chainer implementation](https://github.com/soskek/attention_is_all_you_need).
- Evaluation utilities adapted from [XNMT](https://github.com/neulab/xnmt) and [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
- Flash Attention via PyTorch 2.0 `F.scaled_dot_product_attention`.

---

*This work is part of ongoing research on neural machine translation for low-resource and underrepresented African languages.*
