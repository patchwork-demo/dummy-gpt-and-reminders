import statistics
from pathlib import Path

DATA = "messages_deduped.txt"


def main():
    lines = [
        line.strip()
        for line in Path(DATA).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lengths = sorted(len(line) for line in lines)
    n = len(lengths)
    total = sum(lengths)
    vocab = len(set("".join(lines))) + 1  # unique characters + BOS

    def percentile(p):
        return lengths[min(n - 1, int(n * p / 100))]

    p95 = percentile(95)
    recommended = max(32, ((p95 + 15) // 16) * 16)  # round p95 up to a multiple of 16

    print(f"docs:              {n}")
    print(f"total chars:       {total}")
    print(f"mean len:          {statistics.mean(lengths):.1f}")
    print(f"median len:        {lengths[n // 2]}")
    print(f"p90 / p95 / p99:   {percentile(90)} / {p95} / {percentile(99)}")
    print(f"max len:           {lengths[-1]}")
    print(f"vocab (chars+BOS): {vocab}")
    print(f"recommended --block-size: {recommended}")


if __name__ == "__main__":
    main()
