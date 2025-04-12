import pandas as pd

# Load DailyDialog
daily_df = pd.read_csv("../data/dailydialog_pairs.csv")

# Load cleaned Twitter data (only text, not tokenized)
twitter_df = pd.read_csv("../data/sampled_tokenized_tweets.csv")

# Select and rename relevant columns
twitter_df = twitter_df[["clean_text_customer", "clean_text_company"]].copy()
twitter_df.columns = ["input", "output"]

# Rename DailyDialog columns if needed
daily_df.columns = ["input", "output"]

# Merge both datasets
combined_df = pd.concat([daily_df, twitter_df], ignore_index=True)

# Shuffle the combined data
combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Save to new CSV
combined_df.to_csv("../data/final_conversations.csv", index=False)

print("Final combined dataset saved as final_conversations.csv")
print("Total samples:", len(combined_df))
print(combined_df.head())
