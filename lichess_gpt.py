# Install these dependencies in the terminal/venv before running this .py file:
# sudo apt-get install zstd
# python -m pip install python-chess matplotlib numpy torch
import subprocess
import io
import chess.pgn

N_GAMES = 100_000
OUTPUT_FILE = "games.txt"

# Download August 2026 Lichess database and stream-decompress it.
wget = subprocess.Popen(
    [
        "wget",
        "-qO-",
        "https://database.lichess.org/standard/lichess_db_standard_rated_2026-08.pgn.zst",
    ],
    stdout=subprocess.PIPE,
)

zstd = subprocess.Popen(
    ["zstd", "-d"],
    stdin=wget.stdout,
    stdout=subprocess.PIPE,
)

# Allow wget to receive SIGPIPE if zstd exits early.
wget.stdout.close()

# Convert bytes → text so python-chess can parse PGN.
pgn_stream = io.TextIOWrapper(
    zstd.stdout,
    encoding="utf-8"
)

games_written = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    while games_written < N_GAMES:

        game = chess.pgn.read_game(pgn_stream)

        # End of database
        if game is None:
            break

        # Extract UCI moves
        uci_moves = [
            move.uci()
            for move in game.mainline_moves()
        ]

        # Ignore games with no moves
        if not uci_moves:
            continue

        # One complete game per line
        out.write(" ".join(uci_moves) + "\n")

        games_written += 1

        if games_written % 1000 == 0:
            print(f"Extracted {games_written}/{N_GAMES} games")

pgn_stream.close()

print(f"\nFinished.")
print(f"Games extracted: {games_written}")
print(f"Saved to: {OUTPUT_FILE}")
games = []
unique_uci_moves = set()

with open("games.txt", "r", encoding="utf-8") as f:
    for line in f:
        moves = line.strip().split()

        if moves:
            games.append(moves)
            unique_uci_moves.update(moves)

print("Games:", len(games))
print("Unique UCI moves:", len(unique_uci_moves))
len(games)
import matplotlib.pyplot as plt

# Measure dataset
import numpy as np

total_games = len(games)
total_moves = sum(len(game) for game in games)
lengths = np.array([len(game) for game in games])

print("Total Games: ", total_games)
print("Total moves:", total_moves)
print("Average moves/game:", total_moves / len(games))

game_strings = [
    " ".join(game)
    for game in games
]

unique_games = set(game_strings)

print("Total games:", len(game_strings))
print("Unique games:", len(unique_games))
print("Duplicates:", len(game_strings) - len(unique_games))

duplicate_rate = 1 - len(unique_games) / len(game_strings)

print("Duplicate rate:", duplicate_rate)

# check for variations in first N moves in each game
for n in [5, 10, 20, 30, 40]:
    prefixes = [tuple(game[:n]) for game in games]
    unique_prefixes = len(set(prefixes))

    print(
        f"{n:2d} plies: "
        f"{unique_prefixes:,} unique / {len(games):,} games"
    )

# distribution
print("Min: ", lengths.min())
print("Max: ", lengths.max())
print("Mean: ", lengths.mean())
print("Median: ", np.median(lengths))
print("Std: ", lengths.std())

plt.hist(lengths, bins=50)
plt.xlabel("Moves per game")
plt.ylabel("Number of games")
plt.title("Game length distribution")
plt.show()
# build vocab
# text = "".join(games)
# chars = ['.'] + sorted(set(text))
# vocab_size = len(chars)
# stoi = {ch:i for i,ch in enumerate(chars)}
# itos = {i:ch for i,ch in enumerate(chars)}
# encode = lambda s: [stoi[ch] for ch in s]
# decode = lambda l: "".join(itos[i] for i in l)

# print(encode("e4"))
# print(decode(encode("e4")))
# full move uci tokenizer
vocab = ['.'] + sorted(set(unique_uci_moves))
vocab_size = len(sorted(vocab)) # all the unique moves are the vocabulary
move_to_id = {move:i for i, move in enumerate(vocab)}
id_to_move = {i:move for i, move in enumerate(vocab)}
encode = lambda moves: [move_to_id[move] for move in moves]
decode = lambda ids: ["".join(id_to_move[id] for id in ids)]

print(encode(["e2e4"]))
print(decode(encode(["e2e4"])))
import random
random.seed(42)
random.shuffle(games)

n1 = int(0.8 * len(games))
n2 = int(0.9 * len(games))

train_games = games[:n1]
dev_games = games[n1:n2]
test_games = games[n2:]
def get_batch(games, batch_size, context_size, move_to_id):
    X = torch.zeros(
        (batch_size, context_size),
        dtype=torch.long
    )

    Y = torch.zeros(
        batch_size,
        dtype=torch.long
    )

    for i in range(batch_size):
        game = games[torch.randint(len(games), (1,)).item()]
        pos = torch.randint(len(game), (1,)).item()

        # Only keep the previous context_size moves.
        history = game[max(0, pos - context_size):pos]
        ids = [move_to_id[move] for move in history]

        # Left-pad the context with token 0.
        if ids:
            X[i, -len(ids):] = torch.tensor(ids)

        # Predict the current move.
        Y[i] = move_to_id[game[pos]]

    return X, Y
import torch
import torch.nn as nn
import torch.nn.functional as F

g = torch.Generator().manual_seed(1337)

class MLP(nn.Module):
  def __init__(self, n_embd, h_dim):
    super().__init__()
    self.hidden_layer = nn.Linear(n_embd, h_dim)
    self.relu = nn.ReLU()
    self.output_layer = nn.Linear(h_dim, n_embd)
    self.output_layer = nn.Linear(h_dim, n_embd)

  def forward(self, x):
    x = self.relu(self.hidden_layer(x))
    out = self.output_layer(x)
    return out
import math

class Head(nn.Module):
  def __init__(self, n_embd, head_size, context_size):
    super().__init__()
    self.head_size = head_size
    self.Q = nn.Linear(n_embd, head_size)
    self.K = nn.Linear(n_embd, head_size)
    self.V = nn.Linear(n_embd, head_size)
    self.register_buffer(
        "tril",
        torch.tril(torch.ones(context_size, context_size)) # to match (B, T, T) each example will have T * T attention matrix where each row is a query and each column of that row is the key it needs to attend
    )

  def forward(self, x):
    q = self.Q(x)
    k = self.K(x)
    v = self.V(x)

    s = q @ k.transpose(-2, -1)
    s = s / math.sqrt(self.head_size)

    # causal mask
    mask = self.tril[:x.size(1), :x.size(1)]
    s = s.masked_fill(mask == 0, float("-inf"))

    w = F.softmax(s, dim=-1)

    o = w @ v

    return o
class MHA(nn.Module):
  def __init__(self, n_embd, head_size, n_head, context_size):
    super().__init__()
    self.heads = nn.ModuleList([
        Head(n_embd, head_size, context_size) for _ in range(n_head)
    ])
    self.out_layer = nn.Linear(n_embd, n_embd)

  def forward(self, x):
    outputs = [head(x) for head in self.heads]
    out = torch.cat(outputs, dim=-1)
    out = self.out_layer(out)
    return out
class Block(nn.Module):
  def __init__(self, n_embd, n_head, h_dim, context_size):
    super().__init__()
    head_size = n_embd // n_head
    self.mha = MHA(n_embd, head_size, n_head, context_size)
    self.mlp = MLP(n_embd, h_dim)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x1 = x + self.mha(self.ln1(x))
    x2 = x1 + self.mlp(self.ln2(x1))
    return x2

batch_size=256
n_blocks = 6
n_embd = 384
n_head = 16
h_dim = 4 * n_embd
context_size = 256
class GPT(nn.Module):
  def __init__(self):
    super().__init__()
    self.blocks = nn.ModuleList([
        Block(n_embd, n_head, h_dim, context_size) for _ in range(n_blocks)
    ])
    self.embd_table = nn.Embedding(vocab_size, n_embd)
    self.pos_embd_table = nn.Embedding(context_size, n_embd)
    self.final_ln = nn.LayerNorm(n_embd)
    self.linear = nn.Linear(n_embd, vocab_size)

  def forward(self, x, y=None):
    tok_emb = self.embd_table(x)

    T = tok_emb.shape[1]
    pos_ids = torch.arange(T, device=x.device)
    pos_emb = self.pos_embd_table(pos_ids)

    x = tok_emb + pos_emb

    for block in self.blocks:
      x = block(x)

    x = self.final_ln(x)
    logits = self.linear(x)
    logits = logits[:, -1, :]
    if y is None:
      return logits
    loss = F.cross_entropy(logits, y)
    return logits, loss
# model
model = GPT()
# optimizer
lr = 3e-4
wd = 0.01

total_params = sum(p.numel() for p in model.parameters())
print(total_params)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
# train
batch_size = 32
train_lossi = []
dev_lossi = []
best_dev_loss = float("inf")


@torch.no_grad()
def evaluate(model, games, move_to_id, context_size, batch_size=128):
  """Evaluate loss exhaustively over every move in every game."""
  model.eval()

  total_loss = 0.0
  total_examples = 0

  X_batch = []
  Y_batch = []

  for game in games:
    for pos in range(len(game)):
      # Previous moves available to the model.
      history = game[max(0, pos - context_size):pos]
      context = [0] * context_size
      ids = [move_to_id[move] for move in history]

      if ids:
        context[-len(ids):] = ids

      # Current move is the target.
      target = move_to_id[game[pos]]

      X_batch.append(context)
      Y_batch.append(target)

      if len(X_batch) == batch_size:
        X = torch.tensor(X_batch, dtype=torch.long)
        Y = torch.tensor(Y_batch, dtype=torch.long)

        logits, loss = model(X, Y)

        total_loss += loss.item() * len(Y)
        total_examples += len(Y)

        X_batch = []
        Y_batch = []

  # Evaluate the final partial batch.
  if X_batch:
    X = torch.tensor(X_batch, dtype=torch.long)
    Y = torch.tensor(Y_batch, dtype=torch.long)

    logits, loss = model(X, Y)

    total_loss += loss.item() * len(Y)
    total_examples += len(Y)

  model.train()

  return total_loss / total_examples


for step in range(10000):
  x, y = get_batch(
    train_games,
    batch_size,
    context_size,
    move_to_id
  )

  # forward pass
  logits, loss = model(x, y)

  # backward pass
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  # print loss
  if step % 1000 == 0:
    print(loss.item())

  if step % 500 == 0:
    # Exhaustive evaluation over the complete train and validation splits.
    train_loss = evaluate(
      model,
      train_games,
      move_to_id,
      context_size,
      batch_size=128
    )
    train_lossi.append({
      "step": step,
      "loss": train_loss
    })

    eval_loss = evaluate(
      model,
      dev_games,
      move_to_id,
      context_size,
      batch_size=128
    )
    dev_lossi.append({
      "step": step,
      "loss": eval_loss
    })

    print(
      f"step {step}: "
      f"train loss {train_loss:.4f}, "
      f"eval loss {eval_loss:.4f}"
    )

    if eval_loss < best_dev_loss:
      # Save the best loss as checkpoint.
      torch.save({
        "step": step,
        "eval_loss": eval_loss,
        "model_state_dict": model.state_dict()
      }, "best_checkpoint.pt")

      print(
        f"Saved best checkpoint at step {step}, "
        f"loss {eval_loss:.4f}"
      )
      best_dev_loss = eval_loss
# plot the losses
plt.plot([x["step"] for x in train_lossi], [x["loss"] for x in train_lossi], label="Train Loss")
plt.plot([x["step"] for x in dev_lossi], [x["loss"] for x in dev_lossi], label="Eval Loss")

plt.xlabel("Training steps")
plt.ylabel("Loss")
plt.title("Training loss vs. steps")
plt.legend()
plt.grid(True)
plt.show()
# load the last best checkpoint
checkpoint = torch.load("best_checkpoint.pt")

model.load_state_dict(checkpoint["model_state_dict"])

step = checkpoint["step"]
eval_loss = checkpoint["eval_loss"]

print("Best checkpoint step:", step)
print("Best validation loss:", eval_loss)

# Exhaustive final evaluation of the best checkpoint.
final_train_loss = evaluate(
  model,
  train_games,
  move_to_id,
  context_size,
  batch_size=128
)

final_eval_loss = evaluate(
  model,
  dev_games,
  move_to_id,
  context_size,
  batch_size=128
)

print("Train loss:", final_train_loss)
print("Validation loss:", final_eval_loss)

def generate(max_new_tokens=50):
  context = [0] * context_size
  out = []
  for _ in range(max_new_tokens):
    X = torch.tensor([context])
    logits = model(X)
    probs = F.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    tok_id = next_id.item()
    out.append(tok_id)
    context = context[1:] + [tok_id]

    if tok_id == 0:
      break

  return out

print(decode(generate()))
# -------
# train 10M model on 100K games
# Start the experiment on mech interp.
# -------
# Move the code to vscode
# Write up
