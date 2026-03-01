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

# Source files get a direction tag prepended so the model knows which language
# to generate.  Target files get no tag.
#   <2zh>  →  generate Chinese   (used in Twi→Chi direction)
#   <2tw>  →  generate Twi       (used in Chi→Twi direction)
SRC_TAG = {
    "train.src":    "auto",   # mixed Twi→Chi and Chi→Twi — detect per line
    "val.src":      "<2zh>",  # all Twi sources  → generate Chinese
    "test.src":     "<2zh>",  # all Twi sources  → generate Chinese
    "test_rev.src": "<2tw>",  # all Chinese sources → generate Twi
}


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


def apply_bpe_to_file(sp, path, src_tag=None):
    """BPE-tokenise Twi lines; Chinese lines are left character-tokenised.

    src_tag controls direction-tag prepending for source files:
      'auto'  — detect per line: Twi lines get <2zh>, Chinese lines get <2tw>
      '<2zh>' — prepend <2zh> to every line (all-Twi source files)
      '<2tw>' — prepend <2tw> to every line (all-Chinese source files)
      None    — no tag (target files)
    """
    lines = read_lines(path)
    out, twi_count = [], 0
    for line in lines:
        chinese = is_chinese(line)
        if chinese:
            encoded = line
        else:
            encoded = ' '.join(sp.encode(line, out_type=str))
            twi_count += 1

        if src_tag == 'auto':
            tag = '<2tw>' if chinese else '<2zh>'
            out.append(f"{tag} {encoded}")
        elif src_tag is not None:
            out.append(f"{src_tag} {encoded}")
        else:
            out.append(encoded)

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
    print("\nApplying BPE + direction tags to data files:")
    for fname in DATA_FILES:
        path = os.path.join(DATA_DIR, fname)
        tag  = SRC_TAG.get(fname, None)   # None → target file, no tag
        total, twi = apply_bpe_to_file(sp, path, src_tag=tag)
        tag_info = f"  tag={tag}" if tag else ""
        print(f"  {fname:<22}  {total:6,} lines  ({twi:,} Twi tokenised){tag_info}")

    print(f"\nDone.  SPM model saved at: {MODEL_PREFIX}.model")
    print("Pass  --spm_model data/twi_chi/twi_spm.model  to train.py / translate.py")
