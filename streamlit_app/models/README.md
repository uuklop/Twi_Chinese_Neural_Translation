# Model Files

Place the three trained model artifacts here before running the app.

| File | Description |
|------|-------------|
| `twi_chi_model_best.ckpt` | Trained Transformer checkpoint (best validation BLEU) |
| `twi_chi.vocab.pickle` | Shared vocabulary pickle `{int → token}` |
| `twi_spm.model` | SentencePiece BPE model for Twi tokenization |

These files are produced by the training pipeline in `Attention_is_All_You_Need/`:

```bash
cd ../Attention_is_All_You_Need
bash pipeline.sh train
```

After training completes, copy the artifacts here:

```bash
cp results/twi_chi_model_best.ckpt  ../streamlit_app/models/
cp data/twi_chi/twi_chi.vocab.pickle ../streamlit_app/models/
cp data/twi_chi/twi_spm.model        ../streamlit_app/models/
```

## Custom paths

You can store the model files anywhere and point the app to them via environment variables:

```bash
export MODEL_DIR=/path/to/your/models
# or individually:
export MODEL_CKPT=twi_chi_model_best.ckpt
export MODEL_VOCAB=twi_chi.vocab.pickle
export MODEL_SPM=twi_spm.model
```

> **Note:** These files are excluded from version control (`.gitignore`).
> Never commit large binary files to Git.
