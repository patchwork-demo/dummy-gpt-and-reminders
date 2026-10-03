import argparse
import json
import math
import random

random.seed(67)

parser = argparse.ArgumentParser(description="Train or run a tiny character-level GPT.")
parser.add_argument(
    "--mode",
    choices=["train", "infer"],
    default="train",
    help="'train' fits the model and saves the weights; 'infer' loads saved weights and generates.",
)
parser.add_argument(
    "--model",
    default="model.json",
    help="path to the weights file (saved/loaded as JSON)",
)
parser.add_argument(
    "--steps", type=int, default=3000, help="number of training steps (train mode only)"
)
parser.add_argument("--samples", type=int, default=20, help="number of texts to generate")
parser.add_argument(
    "--temperature", type=float, default=0.5, help="sampling temperature for generation"
)
parser.add_argument(
    "--seed",
    type=int,
    default=67,
    help="random seed for the generation samples (same seed -> same samples)",
)
args = parser.parse_args()


# Let there be Autograd to recursively apply the chain rule through a computation graph
class Value:
    __slots__ = (
        "_children",
        "_local_grads",
        "data",
        "grad",
    )  # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data  # scalar value of this node calculated during forward pass
        self.grad = (
            0  # derivative of the loss w.r.t. this node, calculated in backward pass
        )
        self._children = children  # children of this node in the computation graph
        self._local_grads = (
            local_grads  # local derivative of this node w.r.t. its children
        )

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# Let there be a Tokenizer to translate strings to discrete symbols and back.
# In train mode we build the vocabulary from the dataset; in infer mode it comes from the weights file.
if args.mode == "train":
    # Input dataset `docs`: list[str] of documents (e.g. a dataset of twitch messages)
    with open("messages_deduped.txt", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]

    docs = [d.lower() for d in docs]
    random.shuffle(docs)
    print(f"num docs: {len(docs)}")

    uchars = sorted(
        set("".join(docs))
    )  # unique characters in the dataset become token ids 0..n-1

    # Model hyperparameters, to store the knowledge of the model
    n_layer = 1  # depth of the transformer neural network (number of layers)
    n_embd = 16  # width of the network (embedding dimension)
    block_size = 96  # maximum context length of the attention window (try 64 for longer messages)
    n_head = 4  # number of attention heads
else:
    # Restore vocabulary and hyperparameters saved alongside the weights
    with open(args.model, encoding="utf-8") as f:
        payload = json.load(f)
    uchars = payload["uchars"]
    n_layer = payload["hparams"]["n_layer"]
    n_embd = payload["hparams"]["n_embd"]
    block_size = payload["hparams"]["block_size"]
    n_head = payload["hparams"]["n_head"]

head_dim = n_embd // n_head  # derived dimension of each head
char_to_id = {ch: i for i, ch in enumerate(uchars)}  # O(1) lookup for tokenization
BOS = len(uchars)  # token id for the special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1  # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")


# Initialize the parameters, to store the knowledge of the model
matrix = lambda nout, nin, std=0.08: [
    [Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)
]


def build_state_dict():
    sd = {
        "wte": matrix(vocab_size, n_embd),
        "wpe": matrix(block_size, n_embd),
        "lm_head": matrix(vocab_size, n_embd),
    }
    for i in range(n_layer):
        sd[f"layer{i}.attn_wq"] = matrix(n_embd, n_embd)
        sd[f"layer{i}.attn_wk"] = matrix(n_embd, n_embd)
        sd[f"layer{i}.attn_wv"] = matrix(n_embd, n_embd)
        sd[f"layer{i}.attn_wo"] = matrix(n_embd, n_embd)
        sd[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
        sd[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd)
    return sd


def save_model(path):
    # Store hyperparameters + vocabulary so `--mode infer` can rebuild the exact same model
    data = {
        "hparams": {
            "n_layer": n_layer,
            "n_embd": n_embd,
            "block_size": block_size,
            "n_head": n_head,
        },
        "uchars": uchars,
        "weights": {
            name: [[p.data for p in row] for row in mat]
            for name, mat in state_dict.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


if args.mode == "train":
    state_dict = build_state_dict()
else:
    state_dict = {
        name: [[Value(x) for x in row] for row in mat]
        for name, mat in payload["weights"].items()
    }

params = [
    p for mat in state_dict.values() for row in mat for p in row
]  # flatten params into a single list[Value]
print(f"num params: {len(params)}")


# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    return [sum((wi * xi for wi, xi in zip(wo, x)), Value(0.0)) for wo in w]


def softmax(logits: list[Value]) -> list[Value]:
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps, Value(0.0))
    return [e / total for e in exps]


def rmsnorm(x: list[Value]) -> list[Value]:
    ms = sum((xi * xi for xi in x), Value(0.0)) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def gpt(token_id: int, pos_id: int, keys: list[list[list[Value]]], values: list[list[list[Value]]]) -> list[Value]:
    tok_emb = state_dict["wte"][token_id]  # token embedding
    pos_emb = state_dict["wpe"][pos_id]  # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # joint token and position embedding
    x = rmsnorm(
        x
    )  # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f"layer{li}.attn_wq"])
        k = linear(x, state_dict[f"layer{li}.attn_wk"])
        v = linear(x, state_dict[f"layer{li}.attn_wv"])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs : hs + head_dim]
            k_h = [ki[hs : hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs : hs + head_dim] for vi in values[li]]
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict["lm_head"])
    return logits


if args.mode == "train":
    # Let there be Adam, the blessed optimizer and its buffers
    learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
    m = [0.0] * len(params)  # first moment buffer
    v = [0.0] * len(params)  # second moment buffer

    # Repeat in sequence
    num_steps = args.steps  # number of training steps
    log_every = 1  # print a new log line every N steps (set to 1 to log every step)
    for step in range(num_steps):
        # Take single document, tokenize it, surround it with BOS special token on both sides
        doc = docs[step % len(docs)]
        tokens = [BOS] + [char_to_id[ch] for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        # Forward the token sequence through the model, building up the computation graph all the way to the loss
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        loss = (1 / n) * sum(
            losses, Value(0.0)
        )  # final average loss over the document sequence. May yours be low.

        # Backward the loss, calculating the gradients with respect to all model parameters
        loss.backward()

        # Adam optimizer update: update the model parameters based on the corresponding gradients
        lr_t = learning_rate * (1 - step / num_steps)  # linear learning rate decay
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
            p.grad = 0

        if (step + 1) % log_every == 0 or step == num_steps - 1:
            print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

    # Persist the trained weights so inference can run later without retraining
    save_model(args.model)
    print(f"saved weights to {args.model}")


# Inference: may the model babble back to us
# Reseed here so generation is reproducible and identical whether run after training or via `--mode infer`
random.seed(args.seed)
temperature = args.temperature  # in (0, 1], control the "creativity" of generated text, low to high
print("--- inference (new, hallucinated names) ---")
for sample_idx in range(args.samples):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx + 1:2d}: {''.join(sample)}")
