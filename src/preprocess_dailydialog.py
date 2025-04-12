from datasets import load_dataset
import pandas as pd

# Step 1: Load the full DailyDialog dataset (13k+ dialogues)
dataset = load_dataset("daily_dialog")
sample = dataset["train"]  # ⬅️ Use full dataset

# Step 2: Create input-output pairs from all conversations
inputs = []
outputs = []

for dialogue in sample:
    turns = dialogue["dialog"]
    
    # Pair each utterance with the next
    for i in range(len(turns) - 1):
        input_turn = turns[i].strip()
        response_turn = turns[i + 1].strip()
        if input_turn and response_turn:
            inputs.append(input_turn)
            outputs.append(response_turn)

# Step 3: Create and save as DataFrame
df = pd.DataFrame({"input": inputs, "output": outputs})
df.to_csv("../data/dailydialog_pairs.csv", index=False)

print("✅ Saved full DailyDialog pairs to ../data/dailydialog_pairs.csv")
print(df.head())
print(f"Total pairs: {len(df)}")

