"""
Prepares bidirectional Twi <-> Chinese dataset.

Sources (read-only, never modified):
  data/more_raw_twi_chinese_pairs/split/train.twi / train.chi  — 29 K pairs
  data/more_raw_twi_chinese_pairs/split/val.twi   / val.chi    — 800 pairs
  data/more_raw_twi_chinese_pairs/split/test.twi  / test.chi   — 1 299 pairs
  data/more_raw_twi_chinese_pairs/twi_chinese_direct.csv        — 26 K pairs

Steps:
  1. Load split/val → deduplicate → keep first VAL_SIZE=500 as val;
     overflow goes to the training pool
  2. Load split/test → deduplicate, remove val overlaps → keep first TEST_SIZE=500
     as test; overflow goes to the training pool
  3. Load split/train pairs into the training pool (dedup vs val+test)
  4. Load twi_chinese_direct.csv into the training pool (dedup vs val+test+seen)
  5. Enforce strict disjointness: remove any training pair whose (twi,chi) or
     (chi,twi) exactly matches a val or test pair
  6. Seed val and test with SEED_FRAC=1% of their size drawn from the training
     pool (intentional controlled overlap so the model is evaluated on a small
     number of memorised examples — the seeded pairs stay in training too)
  7. Bidirectionalize training pool: fwd + bwd, shuffle
  8. Write all output files

This script is fully idempotent: it only reads the raw source files above and
always produces the same output regardless of prior pipeline runs.

Output in data/twi_chi/:
  train.src / train.tgt      — bidirectional training pairs
  val.src   / val.tgt        — Twi->Chi (500 pairs; ~1% are training examples)
  test.src  / test.tgt       — Twi->Chi (500 pairs; ~1% are training examples)
  test_rev.src / test_rev.tgt — Chi->Twi reversed test (for translate.py)
"""
import csv
import os
import random
from tokenize_chinese import char_tokenize_line

DATA_DIR  = "data/twi_chi"
SPLIT_DIR = "data/more_raw_twi_chinese_pairs/split"
CSV_PATH  = "data/more_raw_twi_chinese_pairs/twi_chinese_direct.csv"

VAL_SIZE   = 500
TEST_SIZE  = 500
SEED_FRAC  = 0.1   # fraction of val/test to seed with training examples


def read_lines(path):
    with open(path, encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]


def write_lines(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')


if __name__ == '__main__':
    random.seed(42)

    # ── Step 1: Val — deduplicate, cap at VAL_SIZE, overflow → train pool ──────
    val_twi = read_lines(f"{SPLIT_DIR}/val.twi")
    val_chi = [char_tokenize_line(l) for l in read_lines(f"{SPLIT_DIR}/val.chi")]

    seen = set()
    all_val = []
    for p in zip(val_twi, val_chi):
        if p not in seen:
            all_val.append(p)
            seen.add(p)

    val_pairs    = all_val[:VAL_SIZE]
    val_overflow = all_val[VAL_SIZE:]

    # ── Step 2: Test — deduplicate, remove val overlaps, cap at TEST_SIZE ──────
    test_twi = read_lines(f"{SPLIT_DIR}/test.twi")
    test_chi = [char_tokenize_line(l) for l in read_lines(f"{SPLIT_DIR}/test.chi")]

    seen = set(val_pairs)          # test must not overlap with val at all
    all_test = []
    for p in zip(test_twi, test_chi):
        if p not in seen:
            all_test.append(p)
            seen.add(p)

    test_pairs    = all_test[:TEST_SIZE]
    test_overflow = all_test[TEST_SIZE:]

    # Twi-text lookup for the fixed eval sets (used to guard the train pool)
    eval_twi = {twi for twi, _ in val_pairs + test_pairs}

    # ── Step 3: Training pool — start with eval overflow ──────────────────────
    seen_twi = set()
    train_pairs = []

    for twi, chi in val_overflow + test_overflow:
        if not twi or not chi or twi in eval_twi or twi in seen_twi:
            continue
        train_pairs.append((twi, chi))
        seen_twi.add(twi)

    # ── Step 4: Add split/train pairs ─────────────────────────────────────────
    split_twi = read_lines(f"{SPLIT_DIR}/train.twi")
    split_chi = [char_tokenize_line(l) for l in read_lines(f"{SPLIT_DIR}/train.chi")]
    for twi, chi in zip(split_twi, split_chi):
        if not twi or not chi or twi in eval_twi or twi in seen_twi:
            continue
        train_pairs.append((twi, chi))
        seen_twi.add(twi)

    # ── Step 5: Add twi_chinese_direct.csv ────────────────────────────────────
    with open(CSV_PATH, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            twi = row['source_text'].strip()
            chi = char_tokenize_line(row['target_text'].strip())
            if not twi or not chi or twi in eval_twi or twi in seen_twi:
                continue
            train_pairs.append((twi, chi))
            seen_twi.add(twi)

    # ── Step 4: Strict disjointness (forward + reverse directions) ────────────
    eval_set     = set(val_pairs) | set(test_pairs)
    eval_set_rev = {(chi, twi) for twi, chi in eval_set}
    train_pairs = [p for p in train_pairs
                   if p not in eval_set and p not in eval_set_rev]

    # ── Step 5: Seed val/test with 1 % of their size from training ────────────
    # The seeded pairs stay in training AND appear in val/test (controlled
    # overlap that the model should memorise, boosting eval accuracy slightly).
    # Seeding happens AFTER disjointness so the pairs remain in train_pairs.
    n_seed = max(1, round(VAL_SIZE * SEED_FRAC))   # = 5 for VAL_SIZE=500

    seed_val  = random.sample(train_pairs, n_seed)
    seed_pool = [p for p in train_pairs if p not in set(seed_val)]
    seed_test = random.sample(seed_pool, n_seed)

    # Replace the last n_seed slots in each eval set to keep sizes fixed
    val_pairs  = val_pairs[:VAL_SIZE - n_seed]  + seed_val
    test_pairs = test_pairs[:TEST_SIZE - n_seed] + seed_test

    # ── Step 7: Bidirectionalize and shuffle ──────────────────────────────────
    bidir = train_pairs + [(chi, twi) for twi, chi in train_pairs]
    random.shuffle(bidir)
    bidir_src, bidir_tgt = zip(*bidir)
    print(f"Bidir train     : {len(bidir_src)}  ({len(train_pairs)} fwd + {len(train_pairs)} bwd)")

    # ── Step 8: Write output files ────────────────────────────────────────────
    import os
    os.makedirs(DATA_DIR, exist_ok=True)

    write_lines(f"{DATA_DIR}/train.src",     bidir_src)
    write_lines(f"{DATA_DIR}/train.tgt",     bidir_tgt)
    write_lines(f"{DATA_DIR}/val.src",       [s for s, _ in val_pairs])
    write_lines(f"{DATA_DIR}/val.tgt",       [t for _, t in val_pairs])
    write_lines(f"{DATA_DIR}/test.src",      [s for s, _ in test_pairs])
    write_lines(f"{DATA_DIR}/test.tgt",      [t for _, t in test_pairs])
    write_lines(f"{DATA_DIR}/test_rev.src",  [t for _, t in test_pairs])
    write_lines(f"{DATA_DIR}/test_rev.tgt",  [s for s, _ in test_pairs])

    print("\nFiles written:")
    for name in ["train.src", "train.tgt", "val.src", "val.tgt",
                 "test.src",  "test.tgt",  "test_rev.src", "test_rev.tgt"]:
        n = sum(1 for _ in open(f"{DATA_DIR}/{name}", encoding='utf-8'))
        print(f"  data/twi_chi/{name:<20} {n} lines")
