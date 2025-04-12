# src/generate.py
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from model import TransformerChatbot, generate_square_subsequent_mask

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Load model with old config
model = TransformerChatbot(
    vocab_size=30522,
    d_model=128,        # <- IMPORTANT: match old training config
    nhead=8,
    num_layers=2,
    dim_feedforward=512
).to(device)

# Load checkpoint — choose the best epoch
checkpoint_path = "models/epoch_15.pth"  # Or try 13/14 if better
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Input
input_text = "how are you feeling today?"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Start token
tgt_input = torch.tensor([[tokenizer.cls_token_id]], device=device)

# Generation parameters
max_len = 30
top_k = 50
top_p = 0.95
temperature = 1.0

# Top-k and top-p filtering
def top_k_top_p_filtering(logits, top_k=50, top_p=0.95, filter_value=-float("Inf")):
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_vals, kth_idx = torch.topk(logits, top_k, dim=-1)
        min_kth_val = kth_vals[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_kth_val, torch.tensor(filter_value, device=logits.device), logits)

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    for batch_i in range(logits.size(0)):
        indices_to_remove = sorted_indices[batch_i][sorted_indices_to_remove[batch_i]]
        logits[batch_i, indices_to_remove] = filter_value

    return logits

# Generate tokens
for _ in range(max_len):
    tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
    with torch.no_grad():
        output = model(input_ids, tgt_input, tgt_mask=tgt_mask)
        next_token_logits = output[:, -1, :] / temperature
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    tgt_input = torch.cat([tgt_input, next_token], dim=1)

# Decode
generated_text = tokenizer.decode(tgt_input[0], skip_special_tokens=True)
print("💬 Response:", generated_text)
