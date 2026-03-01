#!/usr/bin/env bash
# Full pipeline for bidirectional Twi <-> Chinese translation
# Usage:
#   bash run_twi_chi.sh              # run all stages
#   bash run_twi_chi.sh prepare      # build bidirectional splits + BPE
#   bash run_twi_chi.sh preprocess   # preprocess (run after prepare)
#   bash run_twi_chi.sh train        # train
#   bash run_twi_chi.sh translate    # translate test set (Twi->Chi and Chi->Twi)

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
    # Note: Twi->Chi output is already Chinese characters; post_process_output
    # inside translate.py joins adjacent Chinese chars automatically.
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
    # Note: Chi->Twi output is BPE-tokenised Twi; --spm_model triggers SPM decode
    # inside translate.py so the output file contains readable Twi words.
    echo "Translations saved to results/twi_chi_pred_chi2twi.txt"
fi
