import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Tiny character-level GPT, same architecture as dummy-gpt.py but vectorized.
# The whole sequence is processed in one shot (no Python-level autograd, no
# per-token loops), so it runs on the GPU and uses BLAS/many CPU cores.
# ---------------------------------------------------------------------------


def fmt_hms(seconds):
    """Format a duration in seconds as h:mm:ss (or m:ss under an hour)."""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


parser = argparse.ArgumentParser(
    description="Train or run a tiny character-level GPT in PyTorch."
)
parser.add_argument(
    "--mode",
    choices=["train", "infer"],
    default="train",
    help="'train' fits the model and saves the weights; 'infer' loads saved weights and generates.",
)
parser.add_argument("--model", default="model.pt", help="path to the weights file (torch.save)")
parser.add_argument("--steps", type=int, default=8000, help="number of training steps")
parser.add_argument("--batch-size", type=int, default=32, help="documents per training step")
parser.add_argument("--lr", type=float, default=3e-3, help="peak learning rate")
parser.add_argument(
    "--min-lr", type=float, default=1e-4, help="final learning rate of the cosine decay"
)
parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
parser.add_argument("--warmup", type=int, default=100, help="linear warmup steps")
parser.add_argument(
    "--grad-clip", type=float, default=1.0, help="gradient clipping norm (0 disables)"
)
parser.add_argument(
    "--val-fraction",
    type=float,
    default=0.05,
    help="fraction of documents held out for validation (train only)",
)
parser.add_argument(
    "--val-every", type=int, default=500, help="compute validation loss every N steps"
)
parser.add_argument("--n-layer", type=int, default=2, help="transformer depth (train only)")
parser.add_argument("--n-embd", type=int, default=64, help="embedding width (train only)")
parser.add_argument("--n-head", type=int, default=4, help="attention heads (train only)")
parser.add_argument("--block-size", type=int, default=96, help="context length (train only)")
parser.add_argument(
    "--no-tie-weights",
    dest="tie_weights",
    action="store_false",
    help="disable tying lm_head weights to the token embedding",
)
parser.add_argument("--samples", type=int, default=20, help="number of texts to generate")
parser.add_argument(
    "--temperature", type=float, default=0.5, help="sampling temperature for generation"
)
parser.add_argument(
    "--top-k", type=int, default=0, help="keep only the K most likely tokens (0 = off)"
)
parser.add_argument(
    "--top-p",
    type=float,
    default=1.0,
    help="nucleus sampling: keep tokens up to this cumulative probability (1.0 = off)",
)
parser.add_argument(
    "--repetition-penalty",
    type=float,
    default=1.0,
    help="penalize tokens already in the context; >1 discourages loops/echoing (1.0 = off)",
)
parser.add_argument(
    "--min-tokens",
    type=int,
    default=0,
    help="never emit the end token before this many generated tokens (0 = off)",
)
parser.add_argument(
    "--continuation-only",
    action="store_true",
    help="print only the generated text, without the prompt",
)
parser.add_argument("--prompt", default="", help="seed the generation with this text")
parser.add_argument(
    "--data", default="messages_deduped.txt", help="corpus to train on (train only)"
)
parser.add_argument(
    "--seed", type=int, default=67, help="random seed (same seed -> same samples)"
)
parser.add_argument("--log-every", type=int, default=100, help="print a log line every N steps")
parser.add_argument(
    "--threads", type=int, default=0, help="CPU threads to use (0 = let PyTorch decide)"
)
parser.add_argument(
    "--device",
    default="auto",
    help="'auto', 'cpu', 'cuda', 'xpu', 'mps', ... (default: auto)",
)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

if args.device == "auto":
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"  # Intel GPU (Arc / Iris Xe) via the Level Zero backend
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon
    else:
        device = "cpu"
else:
    device = args.device
if args.threads > 0:
    torch.set_num_threads(args.threads)
print(f"device: {device} | threads: {torch.get_num_threads()}")


# Let there be a Tokenizer to translate strings to discrete symbols and back.
# In train mode we build the vocabulary from the dataset; in infer mode it comes from the weights file.
if args.mode == "train":
    # Input dataset `docs`: list[str] of documents (e.g. a dataset of twitch messages)
    with open(args.data, encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]

    docs = [d.lower() for d in docs]
    random.shuffle(docs)
    print(f"num docs: {len(docs)}")

    uchars = sorted(set("".join(docs)))  # unique characters become token ids 0..n-1

    # Model hyperparameters
    hparams = {
        "n_layer": args.n_layer,  # depth of the transformer neural network
        "n_embd": args.n_embd,  # width of the network (embedding dimension)
        "block_size": args.block_size,  # maximum context length of the attention window
        "n_head": args.n_head,  # number of attention heads
    }
    tie_weights = args.tie_weights
else:
    if not Path(args.model).exists():
        raise SystemExit(
            f"weights file not found: {args.model}\n"
            "Train first (train mode saves the checkpoint), or point --model at an existing one."
        )
    ckpt = torch.load(args.model, map_location="cpu")
    uchars = ckpt["uchars"]
    hparams = ckpt["hparams"]
    tie_weights = ckpt.get("tie_weights", False)

BOS = len(uchars)  # token id for the special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1  # characters + BOS
PAD = vocab_size  # extra id used only to pad batches; never generated
n_classes = vocab_size + 1  # what lm_head predicts: characters, BOS and PAD
char_to_id = {ch: i for i, ch in enumerate(uchars)}
print(f"vocab size: {vocab_size}")

n_layer = hparams["n_layer"]
n_embd = hparams["n_embd"]
block_size = hparams["block_size"]
n_head = hparams["n_head"]


# Define the model architecture: same as dummy-gpt.py, but expressed with torch layers.
# RMSNorm here has a learnable per-channel scale (the pure-Python version had a fixed one).
class RMSNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)


class Attention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)
        # causal mask: a token only attends to itself and the past
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        )

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = Attention(n_embd, n_head, block_size)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, n_classes, n_embd, n_layer, n_head, block_size, tie_weights=True):
        super().__init__()
        self.wte = nn.Embedding(n_classes, n_embd)  # token embedding
        self.wpe = nn.Embedding(block_size, n_embd)  # position embedding
        self.norm = RMSNorm(n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.lm_head = nn.Linear(n_embd, n_classes, bias=False)
        if tie_weights:
            # Share the token embedding with the output projection: fewer parameters and
            # a strong inductive bias that usually helps small models.
            self.lm_head.weight = self.wte.weight

    def forward(self, idx):
        _, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None]  # joint token and position embedding
        x = self.norm(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)


model = GPT(n_classes, n_embd, n_layer, n_head, block_size, tie_weights=tie_weights).to(
    device
)
print(f"num params: {sum(p.numel() for p in model.parameters())}")


if args.mode == "train":
    # Each document -> [BOS] + chars + [BOS], shifted into (input, target) pairs.
    seqs = [[BOS] + [char_to_id[c] for c in d] + [BOS] for d in docs]

    def make_batch(batch):
        xs, ys = [], []
        for s in batch:
            s = s[: block_size + 1]
            x = s[:-1]
            y = s[1:]
            x = x + [PAD] * (block_size - len(x))
            y = y + [PAD] * (block_size - len(y))
            xs.append(x)
            ys.append(y)
        return (
            torch.tensor(xs, device=device),
            torch.tensor(ys, device=device),
        )

    # Hold out a slice for validation so we can tell underfitting from overfitting.
    n_val = int(len(seqs) * args.val_fraction)
    val_seqs = seqs[:n_val]
    train_seqs = seqs[n_val:]
    print(f"train docs: {len(train_seqs)} | val docs: {len(val_seqs)}")

    num_steps = args.steps
    B = args.batch_size

    @torch.no_grad()
    def evaluate(subset):
        model.eval()
        total, count = 0.0, 0
        for i in range(0, len(subset), B):
            x, y = make_batch(subset[i : i + B])
            logits = model(x)
            total += F.cross_entropy(
                logits.reshape(-1, n_classes),
                y.reshape(-1),
                ignore_index=PAD,
                reduction="sum",
            ).item()
            count += int((y != PAD).sum())
        model.train()
        return total / max(1, count)

    # AdamW, the blessed optimizer, with a cosine schedule and gradient clipping
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    def lr_at(step):
        if step < args.warmup:  # linear warmup
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, num_steps - args.warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))  # cosine decay to min_lr
        return args.min_lr + (args.lr - args.min_lr) * cosine

    model.train()
    t0 = time.perf_counter()
    for step in range(num_steps):
        batch = [train_seqs[(step * B + j) % len(train_seqs)] for j in range(B)]
        x, y = make_batch(batch)

        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, n_classes), y.reshape(-1), ignore_index=PAD
        )

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        optimizer.step()

        if (step + 1) % args.log_every == 0 or step == num_steps - 1:
            steps_done = step + 1
            elapsed = time.perf_counter() - t0
            rate = steps_done / elapsed
            eta = elapsed / steps_done * (num_steps - steps_done)
            line = (
                f"step {steps_done:5d} / {num_steps:5d} | loss {loss.item():.4f} | "
                f"lr {lr_at(step):.2e}"
            )
            if val_seqs and ((step + 1) % args.val_every == 0 or step == num_steps - 1):
                line += f" | val {evaluate(val_seqs):.4f}"
            line += f" | {rate:5.1f} st/s | ETA {fmt_hms(eta)}"
            print(line)

    torch.save(
        {
            "hparams": hparams,
            "uchars": uchars,
            "tie_weights": tie_weights,
            "state_dict": model.state_dict(),
        },
        args.model,
    )
    print(f"saved weights to {args.model}")
else:
    model.load_state_dict(ckpt["state_dict"])


def sample_next_token(logits, context, generated_count):
    """Repetition penalty over the context, min length, top-k/top-p, temperature."""
    logits = logits.clone()
    if args.repetition_penalty != 1.0:
        for token_id in set(context):
            if logits[token_id] > 0:
                logits[token_id] /= args.repetition_penalty
            else:
                logits[token_id] *= args.repetition_penalty
    if generated_count < args.min_tokens:
        logits[BOS] = float("-inf")  # do not end the message too early
    logits = logits / args.temperature
    if args.top_k > 0:
        k = min(args.top_k, logits.numel())
        threshold = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if args.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > args.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


# Inference: may the model babble back to us
torch.manual_seed(args.seed)  # reproducible and identical after train or via `--mode infer`
model.eval()
prompt_ids = [char_to_id[ch] for ch in args.prompt if ch in char_to_id]
print("--- inference (new, hallucinated names) ---")
for sample_idx in range(args.samples):
    token_ids = [BOS] + prompt_ids
    generated = []
    with torch.no_grad():
        for _ in range(block_size):
            idx = torch.tensor([token_ids[-block_size:]], device=device)
            logits = model(idx)[0, -1, :vocab_size]  # exclude PAD
            next_id = sample_next_token(logits, token_ids, len(generated))
            if next_id == BOS:
                break
            token_ids.append(next_id)
            generated.append(next_id)
    shown = generated if args.continuation_only else token_ids[1:]
    print(f"sample {sample_idx + 1:2d}: {''.join(uchars[i] for i in shown)}")
