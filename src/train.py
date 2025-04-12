import sys
import os
sys.path.append(os.path.abspath(".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import ElectraModel, ElectraTokenizer  # Use ELECTRA
from src.model import TransformerChatbot, generate_square_subsequent_mask
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# --- Training Config ---
EPOCHS = 6
NHEAD = 8
DMODEL = 256  # Larger model
BATCH_SIZE = 16
MAX_LEN = 128
NUM_LAYERS = 4  # 4-layer Transformer
BEAM_SIZE = 4   # Beam width

# Repetition penalty & n-gram blocking
REPETITION_PENALTY = 1.2
NO_REPEAT_NGRAM_SIZE = 2

# --- Device Setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load ELECTRA model & tokenizer for the source side ---
electra_tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")
electra_model = ElectraModel.from_pretrained("google/electra-base-discriminator").to(device)
electra_model.eval()  # We'll keep it frozen for simplicity

# --- Load dataset ---
df = pd.read_csv("../data/final_conversations.csv")

# We'll still use electra_tokenizer for the source side input (IDs), 
# but the model's decoder vocabulary must match the final fc_out dimension 
# (assumed 30522 here).
def encode_text(txt):
    return electra_tokenizer.encode(
        str(txt),
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LEN
    )

df["input_ids"] = df["input"].apply(encode_text)
df["output_ids"] = df["output"].apply(encode_text)

# --- Padding ---
df["input_ids"] = pad_sequences(df["input_ids"], maxlen=MAX_LEN, padding="post", truncating="post").tolist()
df["output_ids"] = pad_sequences(df["output_ids"], maxlen=MAX_LEN, padding="post", truncating="post").tolist()

# --- Dataset Class ---
class ChatDataset(Dataset):
    def __init__(self, df):
        self.inputs = df["input_ids"].tolist()
        self.targets = df["output_ids"].tolist()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "src": torch.tensor(self.inputs[idx], dtype=torch.long),
            "tgt": torch.tensor(self.targets[idx], dtype=torch.long),
        }

# --- Split dataset ---
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = ChatDataset(train_df)
val_dataset = ChatDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --- Model Setup ---
model = TransformerChatbot(
    d_model=DMODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=512,
    vocab_size=30522  # must match the decoder vocab dimension
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# --- Ensure models folder exists ---
os.makedirs("models", exist_ok=True)
checkpoint_path = "models/latest_checkpoint.pth"
start_epoch = 0

# --- Resume Training from Checkpoint ---
if os.path.exists(checkpoint_path):
    print("Found checkpoint! Loading...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError as e:
            print("⚠️ Optimizer mismatch due to model structure change. Starting with a fresh optimizer.")
            print(str(e))
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch + 1}")
    else:
        print("Checkpoint incompatible. Starting fresh.")

# --- Training Function ---
def train_epoch(model, dataloader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    start_time = time.time()

    loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False)
    for batch in loop:
        src_ids = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        # Split out the input for the decoder
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Generate the causal mask for the decoder side
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

        # 1) Get ELECTRA embeddings for the source
        with torch.no_grad():
            # shape: [B, seq_len, d_model=768], but we'll handle dimension mismatch
            src_embeddings = electra_model(src_ids).last_hidden_state  # [B, seq_len, 768]

        # If ELECTRA's hidden_size = 768 and your model expects d_model=256,
        # we can add a linear projection (if not in model.py).
        # For demonstration, let's do it inline:
        if src_embeddings.size(-1) != DMODEL:
            # Project from 768 -> 256
            # It's typically better to define this projection in the model, but here's a quick inline approach:
            if not hasattr(model, "src_projection"):
                # Create and store a projection layer once
                model.src_projection = nn.Linear(src_embeddings.size(-1), DMODEL).to(device)
            src_embeddings = model.src_projection(src_embeddings)

        # 2) Forward pass
        logits = model(
            src_embeddings,
            tgt_input,
            tgt_mask=tgt_mask
        )

        # 3) Compute loss
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    duration = time.time() - start_time
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1} done in {duration:.2f}s | Avg Loss: {avg_loss:.4f}")
    return avg_loss

# --- Evaluation Function ---
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            src_ids = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

            # ELECTRA embeddings
            src_embeddings = electra_model(src_ids).last_hidden_state
            if src_embeddings.size(-1) != DMODEL:
                src_embeddings = model.src_projection(src_embeddings)

            logits = model(src_embeddings, tgt_input, tgt_mask=tgt_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def seq_contains_ngram(seq, n):
    """
    Check if the last n tokens form a previously used n-gram.
    """
    if len(seq) < n:
        return False
    ngram = tuple(seq[-n:])
    for start_i in range(0, len(seq) - n):
        if tuple(seq[start_i:start_i+n]) == ngram:
            return True
    return False

# --- Beam Search Decoding ---
def beam_search_decode(
    model,
    tokenizer,
    src_ids,
    max_len=MAX_LEN,
    beam_size=BEAM_SIZE,
    repetition_penalty=REPETITION_PENALTY,
    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE
):
    model.eval()
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        src_embeddings = electra_model(src_tensor).last_hidden_state
        if src_embeddings.size(-1) != DMODEL:
            src_embeddings = model.src_projection(src_embeddings)

    # We'll store beams as tuples of (log_prob, tokens)
    beams = [(0.0, [tokenizer.cls_token_id])]

    for _ in range(max_len):
        new_beams = []
        for log_prob, seq in beams:
            if seq[-1] == tokenizer.sep_token_id:
                # Already ended, keep as-is
                new_beams.append((log_prob, seq))
                continue

            tgt_tensor = torch.tensor([seq], dtype=torch.long, device=device)
            tgt_mask = generate_square_subsequent_mask(len(seq)).to(device)

            with torch.no_grad():
                outputs = model(src_embeddings, tgt_tensor, tgt_mask=tgt_mask)

            next_logits = outputs[0, -1, :]

            # Repetition penalty
            for token_id in set(seq):
                if next_logits[token_id] < 0:
                    next_logits[token_id] *= repetition_penalty
                else:
                    next_logits[token_id] /= repetition_penalty

            # Block repeating n-grams
            if no_repeat_ngram_size > 0 and len(seq) >= no_repeat_ngram_size:
                ngram = seq[-(no_repeat_ngram_size-1):]
                for nxt_id in range(next_logits.size(0)):
                    if seq_contains_ngram(seq + [nxt_id], no_repeat_ngram_size):
                        next_logits[nxt_id] = float("-inf")

            probs = torch.softmax(next_logits, dim=-1)
            topk = torch.topk(probs, beam_size)
            for idx in range(beam_size):
                token_id = topk.indices[idx].item()
                token_prob = topk.values[idx].item()
                new_seq = seq + [token_id]
                new_log_prob = log_prob + torch.log(torch.tensor(token_prob + 1e-12)).item()
                new_beams.append((new_log_prob, new_seq))

        # Keep top beam_size
        new_beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
        beams = new_beams

        # If all beams end with [SEP], we can stop early
        if all(b[-1][-1] == tokenizer.sep_token_id for b in beams):
            break

    best_seq = max(beams, key=lambda x: x[0])[1]
    return best_seq

def generate_beam(model, tokenizer, input_text):
    # Encode text with ELECTRA tokenizer
    input_ids = electra_tokenizer.encode(
        input_text, add_special_tokens=True, truncation=True, max_length=MAX_LEN
    )
    out_ids = beam_search_decode(model, electra_tokenizer, input_ids)
    if out_ids and out_ids[0] == electra_tokenizer.cls_token_id:
        out_ids = out_ids[1:]
    if electra_tokenizer.sep_token_id in out_ids:
        out_ids = out_ids[: out_ids.index(electra_tokenizer.sep_token_id)]
    return electra_tokenizer.decode(out_ids, skip_special_tokens=True)

# --- Training Loop ---
train_loss_per_epoch = []
val_loss_per_epoch = []
best_val_loss = float("inf")

for epoch in range(start_epoch, start_epoch + EPOCHS):
    avg_train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch)
    avg_val_loss = evaluate(model, val_loader, criterion)
    train_loss_per_epoch.append(avg_train_loss)
    val_loss_per_epoch.append(avg_val_loss)

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "models/latest_checkpoint.pth")

    torch.save(model.state_dict(), f"models/epoch_{epoch + 1}.pth")
    print(f"Saved checkpoint for epoch {epoch + 1}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "models/best_model.pth")
        print(f"New best model at epoch {epoch+1} - val loss {avg_val_loss:.4f}")

    # Generate sample output using beam search
    sample_input = "How can I reset my password? My account got locked."
    sample_output = generate_beam(model, electra_tokenizer, sample_input)
    print(f"Sample Output [{epoch+1}]: {sample_output}")

# --- Save Loss Curve & Logs ---
plt.plot(range(start_epoch + 1, start_epoch + EPOCHS + 1), train_loss_per_epoch, label='Train Loss')
plt.plot(range(start_epoch + 1, start_epoch + EPOCHS + 1), val_loss_per_epoch, label='Val Loss')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
plt.show()

import pandas as pd
results_df = pd.DataFrame({
    "epoch": list(range(start_epoch + 1, start_epoch + EPOCHS + 1)),
    "train_loss": train_loss_per_epoch,
    "val_loss": val_loss_per_epoch
})
results_df.to_csv("training_log.csv", index=False)

print("Training complete!")
