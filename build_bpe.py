"""
Train SentencePiece BPE on Twi text and apply to all data splits.

Twi lines (non-Chinese) in train.src / train.tgt and all val / test files
are replaced with their BPE-tokenised equivalents.  Chinese lines are left
as-is (already character-tokenised by tokenize_chinese.py).

The SentencePiece model is saved to DATA_DIR/twi_spm.model and is needed
at inference time to detokenise Twi output (see utils.post_process_output).

Run AFTER build_dataset.py and BEFORE preprocess.py.
"""
import os
import sentencepiece as spm

DATA_DIR     = "data/twi_chi"
MODEL_PREFIX = os.path.join(DATA_DIR, "twi_spm")
VOCAB_SIZE   = 4000

DATA_FILES = [
    "train.src", "train.tgt",
    "val.src",   "val.tgt",
    "test.src",  "test.tgt",
    "test_rev.src", "test_rev.tgt",
]


def is_chinese(line):
    return any(
        0x4E00 <= ord(c) <= 0x9FFF
        or 0x3400 <= ord(c) <= 0x4DBF
        or 0xF900 <= ord(c) <= 0xFAFF
        for c in line
    )


def read_lines(path):
    with open(path, encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]


def write_lines(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')


def apply_bpe_to_file(sp, path):
    lines = read_lines(path)
    out, twi_count = [], 0
    for line in lines:
        if is_chinese(line):
            out.append(line)
        else:
            out.append(' '.join(sp.encode(line, out_type=str)))
            twi_count += 1
    write_lines(path, out)
    return len(lines), twi_count


if __name__ == '__main__':
    # Step 1: Collect Twi lines from training data
    train_src = read_lines(f"{DATA_DIR}/train.src")
    train_tgt = read_lines(f"{DATA_DIR}/train.tgt")
    twi_lines = [l for l in train_src + train_tgt if l.strip() and not is_chinese(l)]
    print(f"Collected {len(twi_lines):,} Twi lines for SentencePiece training")

    # Step 2: Train SentencePiece BPE
    spm_input = os.path.join(DATA_DIR, "_twi_for_spm.txt")
    write_lines(spm_input, twi_lines)
    spm.SentencePieceTrainer.train(
        input=spm_input,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        character_coverage=1.0,
        model_type='bpe',
        pad_id=-1, unk_id=0, bos_id=-1, eos_id=-1,
        add_dummy_prefix=False,
    )
    os.remove(spm_input)
    print(f"Trained SPM model → {MODEL_PREFIX}.model  (vocab_size={VOCAB_SIZE})")

    # Step 3: Apply BPE to all data files
    sp = spm.SentencePieceProcessor()
    sp.load(f"{MODEL_PREFIX}.model")
    print("\nApplying BPE to data files:")
    for fname in DATA_FILES:
        path = os.path.join(DATA_DIR, fname)
        total, twi = apply_bpe_to_file(sp, path)
        print(f"  {fname:<22}  {total:6,} lines  ({twi:,} Twi tokenised)")

    print(f"\nDone.  SPM model saved at: {MODEL_PREFIX}.model")
    print("Pass  --spm_model data/twi_chi/twi_spm.model  to train.py / translate.py")
