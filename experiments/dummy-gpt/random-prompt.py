#!/usr/bin/env python3
import argparse
import random
import subprocess
import sys
from pathlib import Path


def load_lines(path):
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def pick_word(lines, min_len, max_len):
    """Return a random word (letters/digits/underscore) from the corpus."""
    words = []
    for line in lines:
        for raw in line.split():
            word = "".join(ch for ch in raw if ch.isalnum() or ch == "_")
            if min_len <= len(word) <= max_len:
                words.append(word)
    return random.choice(words) if words else "привет"


def pick_phrase(lines, min_words, max_words):
    """Return a random contiguous span of words from a random line."""
    chunks = [line.split() for line in lines]
    chunks = [parts for parts in chunks if len(parts) >= min_words]
    if not chunks:
        return random.choice(lines) if lines else "привет как дела"
    parts = random.choice(chunks)
    count = random.randint(min_words, min(max_words, len(parts)))
    start = random.randint(0, len(parts) - count)
    return " ".join(parts[start : start + count])


def main():
    parser = argparse.ArgumentParser(
        description="Pick a random word or phrase from the corpus and run generation with it "
        "as --prompt. Everything you don't recognize here is forwarded to the generator."
    )
    parser.add_argument("--corpus", default="messages_deduped.txt", help="word/phrase source")
    parser.add_argument("--gen", default="dummy-gpt-bpe.py", help="generator script")
    parser.add_argument("--min-len", type=int, default=3, help="minimum word length")
    parser.add_argument("--max-len", type=int, default=20, help="maximum word length")
    parser.add_argument(
        "--phrase-prob",
        type=float,
        default=0.5,
        help="probability of using a phrase instead of a single word",
    )
    parser.add_argument("--phrase-min-words", type=int, default=2)
    parser.add_argument("--phrase-max-words", type=int, default=5)
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="include the seed prompt in each generated sample",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=10,
        help="force at least this many generated tokens (avoids stopping immediately)",
    )
    args, extra = parser.parse_known_args()

    lines = load_lines(args.corpus)
    if random.random() < args.phrase_prob:
        prompt = pick_phrase(lines, args.phrase_min_words, args.phrase_max_words)
        kind = "phrase"
    else:
        prompt = pick_word(lines, args.min_len, args.max_len)
        kind = "word"
    print(f"prompt ({kind}): {prompt}")

    cmd = [
        "uv",
        "run",
        "--extra",
        "torch",
        args.gen,
        "--mode",
        "infer",
        "--prompt",
        prompt,
        "--min-tokens",
        str(args.min_tokens),
    ]
    if not args.show_prompt:
        cmd.append("--continuation-only")
    cmd.extend(extra)
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
