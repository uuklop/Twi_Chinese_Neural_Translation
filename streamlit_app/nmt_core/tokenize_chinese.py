"""
Converts Chinese text files to character-level tokenization.
Each Chinese character becomes a separate space-delimited token.
Usage: python chi_tokenize.py <input_file> <output_file>
"""
import sys
import unicodedata


def is_chinese_char(ch):
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF   # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
        or 0x20000 <= cp <= 0x2A6DF  # CJK Extension B
        or 0xF900 <= cp <= 0xFAFF  # CJK Compatibility Ideographs
        or 0x2F800 <= cp <= 0x2FA1F  # CJK Compatibility Supplement
    )


def char_tokenize_line(line):
    tokens = []
    for ch in line.strip():
        if ch == ' ':
            continue           # drop existing phrase-level spaces
        if is_chinese_char(ch):
            tokens.append(ch)  # each Chinese character is its own token
        else:
            tokens.append(ch)  # punctuation / digits kept as-is
    return ' '.join(tokens)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python chi_tokenize.py <input_file> <output_file>")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path, encoding='utf-8') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            fout.write(char_tokenize_line(line) + '\n')

    print(f"Tokenized {in_path} -> {out_path}")
