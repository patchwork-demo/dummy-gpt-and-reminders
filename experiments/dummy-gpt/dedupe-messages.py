import argparse
import re

# Strict (default) preprocessing: lowercase, normalize mentions, keep only a whitelist
# of characters and drop duplicate lines.
KEEP = re.compile(r"[^a-zа-яё0-9 .,!?:;'\"()\-+=*/@#%&\[\]<>_^]")

# Twitch usernames (Latin letters, digits, underscore) after an @.
MENTION = re.compile(r"@[a-z0-9_]+")


def clean_strict(line: str) -> str:
    line = line.strip().lower()
    if "http" in line:  # drop messages that are mostly long Discord URLs
        return ""
    line = MENTION.sub("@user", line)
    return KEEP.sub("", line)


def clean_shitpost(line: str) -> str:
    # Keep almost everything: case (CAPS = shouting), emoji, urls, mentions and
    # repeated spam. Only tidy whitespace.
    return line.strip().replace("\t", " ")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean messages.txt into a training corpus."
    )
    parser.add_argument("--input", default="messages.txt", help="raw messages file")
    parser.add_argument(
        "--output", default="messages_deduped.txt", help="cleaned output file"
    )
    parser.add_argument(
        "--shitpost",
        action="store_true",
        help="keep case, emoji, urls and duplicates (chaotic data for a shitposting bot)",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        lines = f.readlines()

    clean = clean_shitpost if args.shitpost else clean_strict
    dedupe = not args.shitpost  # keep repeats/spam in shitpost mode

    seen = set()
    out = []
    for line in lines:
        cleaned = clean(line)
        if not cleaned:
            continue
        if dedupe:
            if cleaned in seen:
                continue
            seen.add(cleaned)
        out.append(cleaned + "\n")

    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(out)

    print(f"Source length: {len(lines)}; Output length: {len(out)}")


if __name__ == "__main__":
    main()
