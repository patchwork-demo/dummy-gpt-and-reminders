import argparse
import glob
import json


def load_comments(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data, data.get("comments", [])
    return None, data  # a bare list of comments


def main():
    parser = argparse.ArgumentParser(
        description="Merge several TwitchDownloader chat JSON files into one."
    )
    parser.add_argument("inputs", nargs="+", help="chat JSON files or glob patterns")
    parser.add_argument("-o", "--output", default="chat.json", help="merged output file")
    args = parser.parse_args()

    files = []
    for pattern in args.inputs:
        files.extend(sorted(glob.glob(pattern)) or [pattern])

    merged_meta = None
    comments = []
    seen = set()
    for path in files:
        meta, items = load_comments(path)
        if merged_meta is None and meta is not None:
            merged_meta = meta  # keep the metadata block of the first file
        for comment in items:
            key = comment.get("_id") if isinstance(comment, dict) else None
            if key is not None:
                if key in seen:
                    continue
                seen.add(key)
            comments.append(comment)

    out = dict(merged_meta) if merged_meta is not None else {}
    out["comments"] = comments

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"merged {len(files)} file(s) -> {len(comments)} comments in {args.output}")


if __name__ == "__main__":
    main()
