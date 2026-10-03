import argparse
import glob
import re
import sys
from pathlib import Path

MASTER = "vod-urls.txt"
ID = re.compile(r"\d{6,}")


def read_inputs(patterns):
    """Return the concatenated text of the given files (or stdin)."""
    if not patterns:
        return sys.stdin.read()
    chunks = []
    for pattern in patterns:
        if pattern == "-":
            chunks.append(sys.stdin.read())
            continue
        for path in sorted(glob.glob(pattern)) or [pattern]:
            chunks.append(Path(path).read_text(encoding="utf-8"))
    return "".join(chunks)


def main():
    parser = argparse.ArgumentParser(
        description="Append new VOD ids (from URLs or bare ids) to the master list."
    )
    parser.add_argument(
        "inputs", nargs="*", help="files with URLs/ids; '-' or none means stdin"
    )
    parser.add_argument("--master", default=MASTER, help="master list file")
    args = parser.parse_args()

    master = Path(args.master)
    if master.exists():
        existing = set(ID.findall(master.read_text(encoding="utf-8")))
    else:
        existing = set()

    new = []
    for video_id in ID.findall(read_inputs(args.inputs)):
        if video_id not in existing:
            existing.add(video_id)
            new.append(video_id)

    if new:
        with master.open("a", encoding="utf-8") as f:
            for video_id in new:
                f.write(f"https://www.twitch.tv/videos/{video_id}\n")

    print(f"added {len(new)} new id(s); master now has {len(existing)}")


if __name__ == "__main__":
    main()
