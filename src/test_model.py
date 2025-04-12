import torch
from src.model import TransformerChatbot, generate_square_subsequent_mask

# Define dummy parameters
vocab_size = 30522  # Same as BERT base
batch_size = 2
src_len = 10
tgt_len = 8

# Create random input tensors
src = torch.randint(0, vocab_size, (batch_size, src_len))  # shape: [batch, src_len]
tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))  # shape: [batch, tgt_len]

# Create attention mask for decoder
tgt_mask = generate_square_subsequent_mask(tgt_len)

# Instantiate the model
model = TransformerChatbot(vocab_size)

# Forward pass
output = model(src, tgt, tgt_mask=tgt_mask)

print("✅ Model output shape:", output.shape)
