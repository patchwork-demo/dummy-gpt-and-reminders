# dummy-gpt-torch — quick guide

A tiny character-level GPT written in PyTorch. It trains on `messages_deduped.txt`,
saves the weights, and then generates new messages.

## Setup

Run from the project root (uses the project `.venv`, Python 3.14):

```sh
uv sync --extra torch
```

`torch` is an optional dependency, so it is only installed with `--extra torch`.

## Run

Train (default mode), save weights to `model.pt`, then generate:

```sh
uv run --extra torch dummy-gpt-torch.py
```

Generate from previously saved weights (no dataset needed, no training):

```sh
uv run --extra torch dummy-gpt-torch.py --mode infer --model model.pt
```

Training and inference are reproducible: the same `--seed` yields the same samples.

## Parameters

Architecture flags are used in `train` mode only. In `infer` mode the architecture,
vocabulary and tying setting are read from the checkpoint, so those flags are ignored.

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `train` | `train` fits + saves; `infer` loads + generates |
| `--model` | `model.pt` | path to the weights file (`torch.save`) |
| `--steps` | `8000` | training steps |
| `--batch-size` | `32` | documents per training step |
| `--lr` / `--min-lr` | `3e-3` / `1e-4` | peak / final LR of the cosine decay |
| `--weight-decay` | `0.01` | AdamW weight decay |
| `--warmup` | `100` | linear LR warmup steps |
| `--grad-clip` | `1.0` | gradient clipping norm (`0` disables) |
| `--val-fraction` | `0.05` | fraction of documents held out for validation |
| `--val-every` | `500` | compute validation loss every N steps |
| `--n-layer` | `2` | transformer depth |
| `--n-embd` | `64` | embedding width |
| `--n-head` | `4` | attention heads (`n-embd` must divide by it) |
| `--block-size` | `96` | context length (max message length seen) |
| `--no-tie-weights` | tied | disable tying `lm_head` to the token embedding |
| `--samples` | `20` | number of messages to generate |
| `--temperature` | `0.5` | sampling temperature (`0.5–0.7` is a good range) |
| `--seed` | `67` | random seed |
| `--log-every` | `100` | print a log line every N steps |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `xpu`, `mps`, ... |
| `--threads` | `0` | CPU threads to use (`0` = let PyTorch decide) |

How to tune, in order of impact:

1. **More data** first (see below) — it matters more than any hyperparameter.
2. **Train longer / bigger**: `--steps 40000 --n-embd 128 --n-layer 3`. Watch `loss`
   (train) and `val` (validation):
   - both still falling → train longer or make the model bigger;
   - train falling while `val` rises → overfitting, add data or shrink the model.
3. **`--temperature`**: lower = safer and more repetitive, higher = more creative
   but more nonsense.
4. **`--block-size`**: set it at least to the p90 of your message lengths, or long
   messages get truncated. Compute the p90 from `messages_deduped.txt` if unsure.

A small validation slice (`--val-fraction`, 5% by default) is held out from training;
`val` loss is printed alongside `loss` every `--val-every` steps. Use it — not train
loss — to decide whether you are underfitting or overfitting.

## Recipes (pick one)

> Always pass `--n-embd` and `--n-layer` explicitly. If you omit them the defaults
> (`64` / `2`, only ~109k parameters) are used — it is easy to train a much smaller
> model than you intended by accident.

All recipes below assume `--threads 8` (drop it to let PyTorch auto-detect).
Parameters are for the current 96-token vocabulary.

| Recipe | Key flags | Params | ~Epochs | Rough time on CPU |
|---|---|---|---|---|
| Small / quick | `--n-embd 64 --n-layer 2 --block-size 64 --steps 20000 --batch-size 64` | ~109k | ~22 | ~20 min |
| Recommended | `--n-embd 128 --n-layer 3 --block-size 96 --steps 20000 --batch-size 64` | ~615k | ~22 | ~1.5 h |
| Big / best | `--n-embd 128 --n-layer 3 --block-size 96 --steps 40000 --batch-size 64` | ~615k | ~44 | ~3 h |

Recommended:

```sh
uv run --extra torch dummy-gpt-torch.py \
    --n-embd 128 --n-layer 3 --block-size 96 \
    --steps 20000 --batch-size 64 --threads 8
```

Notes:

- Times are approximate; the training log prints `st/s` (steps per second) and an
  `ETA`, so read the real numbers from your run.
- The cheap speed knobs are `--block-size` (attention is ~quadratic in it) and
  `--steps`. `--batch-size` mostly changes efficiency, not how much data you see per
  epoch — bigger batches give bigger matmuls and better CPU use.
- `~Epochs` = `steps * batch-size / train docs` (the 5% validation split is excluded).

## Dataset size

Input pipeline:

```sh
chat.json --(chat-message-body-to-txt.py)--> messages.txt --(dedupe-messages.py)--> messages_deduped.txt
```

The current dataset: **83,150 raw comments → 60,180 after cleaning,
~1,846,000 characters, vocab 96** (42 VODs merged into one `chat.json`).

Recommended amount (char-level, for Russian chat messages):

| Messages | Expected output |
|---|---|
| < 5,000 | mostly gibberish / random letter soup |
| ~5,000–20,000 | recognizable words and short fragments |
| ~50,000 | plausible words, some coherent short phrases |
| ~100,000+ | smoother, more sentence-like text |

So aim for **at least ~50,000 messages** for decent results, and **100,000+** if you
want visibly better text. The dataset is now in the ~50k–100k band, so quality is
mainly limited by how long you train and how big the model is. Add more chat logs
(more streams / channels) and re-run the pipeline above to grow it further.

## Full pipeline (Twitch -> training)

End-to-end flow, from collecting VOD links in the browser to a trained model:

```sh
# 1. Collect VOD links.
#    Open a Twitch page that lists videos (e.g. a channel's "Videos" tab), open
#    DevTools -> Console and run the contents of collect-twitch-vod-links.js.
#    It scrolls the page, lists every video and copies the URLs to the clipboard.
#    Save them, one per line, to vod-urls.txt.

# 2. Download the chat of every VOD and merge it into a single chat.json.
./fetch-twitch-chats.sh vod-urls.txt
#    -> writes chats/<videoId>.json per VOD (already-downloaded ones are skipped)
#    -> merges them into chat.json (comments deduplicated by _id)

# 3. Extract message bodies and clean them.
python3 chat-message-body-to-txt.py     # chat.json -> messages.txt
python3 dedupe-messages.py              #            -> messages_deduped.txt

# 4. Train and generate (see "Recipes" above for other profiles).
uv run --extra torch dummy-gpt-torch.py \
    --n-embd 128 --n-layer 3 --block-size 96 \
    --steps 20000 --batch-size 64 --threads 8
```

Notes:

- `fetch-twitch-chats.sh` is configurable via env vars: `TWITCH_DOWNLOADER` (path to
  `TwitchDownloaderCLI`), `RAW_DIR` (per-VOD JSON folder, default `chats`),
  `OUT` (merged file, default `chat.json`).
- Re-running step 2 is safe: existing `chats/<videoId>.json` files are skipped, and a
  failed VOD does not stop the rest.
- `chat-message-body-to-txt.py` only uses the `comments[].message.body` field of
  `chat.json`, so the merge metadata (which keeps the first VOD's `video` block) does
  not matter for training.
