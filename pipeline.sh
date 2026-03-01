#!/usr/bin/env bash
# Full pipeline for bidirectional Twi <-> Chinese translation
# Usage:
#   bash pipeline.sh              # run all stages
#   bash pipeline.sh prepare      # build bidirectional splits + BPE
#   bash pipeline.sh preprocess   # preprocess (run after prepare)
#   bash pipeline.sh train        # train
#   bash pipeline.sh translate    # translate test set (Twi->Chi and Chi->Twi)
#   bash pipeline.sh eval         # compute BLEU on translated output
#   bash pipeline.sh plot         # plot training curves
#   bash pipeline.sh tensorboard  # launch TensorBoard

set -e

DATA_DIR="data/twi_chi"
SAVE_DATA="twi_chi"
SPM_MODEL="$DATA_DIR/twi_spm.model"
GPU=0           # set to -1 to use CPU

STAGE=${1:-all}

# ─── Step 1: Build bidirectional splits ───────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "prepare" ]]; then
    echo "=== Step 1a: Building bidirectional splits (val=500, test=500) ==="
    python build_dataset.py

    echo "=== Step 1b: Training SentencePiece BPE on Twi + applying to all splits ==="
    python build_bpe.py
fi

# ─── Step 2: Preprocess ────────────────────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "preprocess" ]]; then
    echo "=== Step 2: Preprocessing ==="
    python preprocess.py \
        -i "$DATA_DIR" \
        -s-train train.src \
        -t-train train.tgt \
        -s-valid val.src \
        -t-valid val.tgt \
        -s-test  test.src \
        -t-test  test.tgt \
        --save_data "$SAVE_DATA"
fi

# ─── Step 3: Train ─────────────────────────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
    echo "=== Step 3: Training ==="
    mkdir -p results
    python train.py \
        -i "$DATA_DIR" \
        --data "$SAVE_DATA" \
        --wbatchsize 4096 \
        --batchsize 60 \
        --tied \
        --beam_size 5 \
        --epoch 60 \
        --layers 6 \
        --multi_heads 8 \
        --warmup_steps 4000 \
        --gpu "$GPU" \
        --out results \
        --model_file      results/twi_chi_model.ckpt \
        --best_model_file results/twi_chi_model_best.ckpt \
        --dev_hyp         results/twi_chi_valid.out \
        --dev_hyp_rev     results/twi_chi_valid_rev.out \
        --test_hyp        results/twi_chi_test.out \
        --test_hyp_rev    results/twi_chi_test_rev.out \
        --spm_model       "$SPM_MODEL"
fi

# ─── Step 4: Translate ─────────────────────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "translate" ]]; then
    echo "=== Step 4a: Translating test set (Twi->Chi) ==="
    python translate.py \
        -i "$DATA_DIR" \
        --data "$SAVE_DATA" \
        --batchsize 60 \
        --beam_size 5 \
        --best_model_file results/twi_chi_model_best.ckpt \
        --model_file      results/twi_chi_model.ckpt \
        --src             "$DATA_DIR/test.src" \
        --output          results/twi_chi_pred_twi2chi.txt \
        --gpu "$GPU"
    echo "Translations saved to results/twi_chi_pred_twi2chi.txt"

    echo "=== Step 4b: Translating test set (Chi->Twi) ==="
    python translate.py \
        -i "$DATA_DIR" \
        --data "$SAVE_DATA" \
        --batchsize 60 \
        --beam_size 5 \
        --best_model_file results/twi_chi_model_best.ckpt \
        --model_file      results/twi_chi_model.ckpt \
        --src             "$DATA_DIR/test_rev.src" \
        --output          results/twi_chi_pred_chi2twi.txt \
        --spm_model       "$SPM_MODEL" \
        --gpu "$GPU"
    echo "Translations saved to results/twi_chi_pred_chi2twi.txt"
fi

# ─── Step 5: Evaluate (BLEU) ───────────────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "eval" ]]; then
    echo "=== Step 5: BLEU Evaluation ==="
    python - <<PYEOF
import sys, os
sys.path.insert(0, '.')
from metrics import BLEUEvaluator
import sentencepiece as spm

def read_lines(path):
    with open(path, encoding='utf-8') as f:
        return [l.rstrip('\n') for l in f]

# ── Twi→Chi: character-level BLEU ─────────────────────────────────────────────
# Reference (test.tgt) is already char-tokenised (space-separated chars).
# Hypothesis has adjacent chars joined by post_process_output; re-split to chars.
ref_chi = [list(l.replace(' ', '')) for l in read_lines('$DATA_DIR/test.tgt')]
hyp_chi = [list(l.replace(' ', '')) for l in read_lines('results/twi_chi_pred_twi2chi.txt')]
bleu_fwd = BLEUEvaluator().evaluate(ref_chi, hyp_chi)
print(f'Twi -> Chi  BLEU: {bleu_fwd}')

# ── Chi→Twi: word-level BLEU (SPM-decode the BPE reference) ──────────────────
sp = spm.SentencePieceProcessor()
sp.load('$SPM_MODEL')

ref_twi_bpe = read_lines('$DATA_DIR/test_rev.tgt')
ref_twi = [sp.decode(l.split()).split() for l in ref_twi_bpe]
hyp_twi = [l.split() for l in read_lines('results/twi_chi_pred_chi2twi.txt')]
bleu_rev = BLEUEvaluator().evaluate(ref_twi, hyp_twi)
print(f'Chi -> Twi  BLEU: {bleu_rev}')

fwd_val = (bleu_fwd.bleu or 0.0) * 100
rev_val = (bleu_rev.bleu or 0.0) * 100
print(f'Average     BLEU: {(fwd_val + rev_val) / 2:.4f}')
PYEOF
fi

# ─── Step 6: Plot training curves ─────────────────────────────────────────────
if [[ "$STAGE" == "plot" ]]; then
    echo "=== Step 6: Plotting training curves ==="
    python plot_training.py --metrics results/metrics.jsonl
fi

# ─── TensorBoard ──────────────────────────────────────────────────────────────
if [[ "$STAGE" == "tensorboard" ]]; then
    echo "=== Launching TensorBoard ==="
    echo "Open http://localhost:6006 in your browser"
    tensorboard --logdir results/runs
fi
