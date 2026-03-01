"""
Prepares bidirectional Twi <-> Chinese dataset.

Steps:
  1. Recover original Twi->Chi pairs from existing bidirectional train.src/tgt
     (detect direction by presence of CJK characters in source)
  2. Include current val.src/tgt and test.src/tgt pairs
  3. Merge in new pairs from twi_chinese_direct.csv (char-tokenize Chinese side)
  4. Deduplicate on Twi source text
  5. Shuffle and split: val=500, test=500, train=rest
  6. Build bidirectional train (Twi->Chi + Chi->Twi, shuffled)
  7. Enforce strict train/val/test disjointness
  8. Write all output files

Output files in data/twi_chi/:
  train.src / train.tgt  — bidirectional training pairs
  val.src   / val.tgt    — Twi->Chi only (500 pairs)
  test.src  / test.tgt   — Twi->Chi only (500 pairs)
  test_rev.src / test_rev.tgt  — Chi->Twi (for translate.py)
"""
import csv
import random
from tokenize_chinese import char_tokenize_line

DATA_DIR = "data/twi_chi"
CSV_PATH = "data/more_raw_twi_chinese_pairs/twi_chinese_direct.csv"


def read_lines(path):
    with open(path, encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]


def write_lines(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')


def is_chinese(line):
    return any(0x4E00 <= ord(c) <= 0x9FFF
               or 0x3400 <= ord(c) <= 0x4DBF
               or 0xF900 <= ord(c) <= 0xFAFF
               for c in line)


if __name__ == '__main__':
    random.seed(42)

    # Step 1: Recover Twi->Chi forward pairs from bidirectional train
    train_src_lines = read_lines(f"{DATA_DIR}/train.src")
    train_tgt_lines = read_lines(f"{DATA_DIR}/train.tgt")
    existing_pairs = []
    for src, tgt in zip(train_src_lines, train_tgt_lines):
        if not is_chinese(src):
            existing_pairs.append((src, tgt))
    print(f"Recovered from train  : {len(existing_pairs)} pairs")

    # Step 2: Add current val and test (already Twi->Chi, chi char-tokenised)
    val_pairs  = list(zip(read_lines(f"{DATA_DIR}/val.src"),
                          read_lines(f"{DATA_DIR}/val.tgt")))
    test_pairs = list(zip(read_lines(f"{DATA_DIR}/test.src"),
                          read_lines(f"{DATA_DIR}/test.tgt")))
    existing_pairs += val_pairs + test_pairs
    print(f"After adding val+test : {len(existing_pairs)} pairs")

    # Step 3: Merge new CSV pairs (char-tokenise Chinese side)
    seen_twi = {src for src, _ in existing_pairs}
    new_pairs, skipped = [], 0
    with open(CSV_PATH, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            twi = row['source_text'].strip()
            chi = char_tokenize_line(row['target_text'].strip())
            if not twi or not chi or twi in seen_twi:
                skipped += 1
                continue
            new_pairs.append((twi, chi))
            seen_twi.add(twi)
    print(f"New CSV pairs added   : {len(new_pairs)}  (skipped duplicates/empty: {skipped})")

    # Step 4: Pool and shuffle
    all_pairs = existing_pairs + new_pairs
    random.shuffle(all_pairs)
    print(f"Total unique pairs    : {len(all_pairs)}")

    # Step 5: Split
    VAL_SIZE, TEST_SIZE = 500, 500
    new_val   = all_pairs[:VAL_SIZE]
    new_test  = all_pairs[VAL_SIZE:VAL_SIZE + TEST_SIZE]
    new_train = all_pairs[VAL_SIZE + TEST_SIZE:]

    # Step 6: Enforce strict disjointness
    eval_set     = set(new_val) | set(new_test)
    eval_set_rev = {(t, s) for s, t in eval_set}
    before    = len(new_train)
    new_train = [p for p in new_train if p not in eval_set and p not in eval_set_rev]
    removed   = before - len(new_train)
    if removed:
        print(f"Disjointness filter   : removed {removed} pairs from train")
    print(f"Split  →  train: {len(new_train)}, val: {len(new_val)}, test: {len(new_test)}")

    # Step 7: Bidirectionalize train
    bidir = list(new_train) + [(tgt, src) for src, tgt in new_train]
    random.shuffle(bidir)
    bidir_src, bidir_tgt = zip(*bidir)
    print(f"Bidir train pairs     : {len(bidir_src)}  ({len(new_train)} fwd + {len(new_train)} bwd)")

    # Step 8: Write output files
    write_lines(f"{DATA_DIR}/train.src", bidir_src)
    write_lines(f"{DATA_DIR}/train.tgt", bidir_tgt)
    write_lines(f"{DATA_DIR}/val.src",      [s for s, _ in new_val])
    write_lines(f"{DATA_DIR}/val.tgt",      [t for _, t in new_val])
    write_lines(f"{DATA_DIR}/test.src",     [s for s, _ in new_test])
    write_lines(f"{DATA_DIR}/test.tgt",     [t for _, t in new_test])
    write_lines(f"{DATA_DIR}/test_rev.src", [t for _, t in new_test])
    write_lines(f"{DATA_DIR}/test_rev.tgt", [s for s, _ in new_test])

    print("\nFiles written:")
    for name in ["train.src", "train.tgt", "val.src", "val.tgt",
                 "test.src",  "test.tgt",  "test_rev.src", "test_rev.tgt"]:
        with open(f"{DATA_DIR}/{name}", encoding='utf-8') as f:
            n = sum(1 for _ in f)
        print(f"  data/twi_chi/{name:<20} {n} lines")
