import pandas as pd
import re
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm  # Progress tracking for tokenization

# Load dataset
file = "/content/drive/MyDrive/Mini-LLM/data/customer_support_tweets.csv"
df = pd.read_csv(file)

print(df.info())
print(df.head())

# Fix 'response_tweet_id' conversion issue
df["response_tweet_id"] = df["response_tweet_id"].astype(str).apply(lambda x: x.split(",")[0] if "," in x else x)
df["response_tweet_id"] = pd.to_numeric(df["response_tweet_id"], errors="coerce").fillna(0).astype("Int64")

# Clean text function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"@\w+", "", text)  # Remove @mentions
    text = re.sub(r"[^A-Za-z0-9@\s]", "", text)  # Remove other special chars
    return text.lower().strip()

df["clean_text"] = df["text"].apply(clean_text)

print(df[["text", "clean_text"]].head())

# Separate customer queries and company responses
customer_tweets = df[df["inbound"] == True].copy()
company_responses = df[df["inbound"] == False].copy()

# Convert IDs to Int64 to avoid warnings
customer_tweets.loc[:, "response_tweet_id"] = customer_tweets["response_tweet_id"].astype("Int64")
company_responses.loc[:, "tweet_id"] = company_responses["tweet_id"].astype("Int64")

# Merge based on response_tweet_id
conversation_df = customer_tweets.merge(
    company_responses, left_on="response_tweet_id", right_on="tweet_id", suffixes=("_customer", "_company")
)

conversation_df = conversation_df[["clean_text_customer", "clean_text_company"]]

print(conversation_df.head())

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Use tqdm for progress tracking
tqdm.pandas()

# Tokenize text with progress tracking
conversation_df["customer_tokens"] = conversation_df["clean_text_customer"].progress_apply(
    lambda x: tokenizer.encode(x, add_special_tokens=True)
)
conversation_df["company_tokens"] = conversation_df["clean_text_company"].progress_apply(
    lambda x: tokenizer.encode(x, add_special_tokens=True)
)

# Padding
MAX_LEN = 128
conversation_df["customer_tokens"] = pad_sequences(
    conversation_df["customer_tokens"], maxlen=MAX_LEN, padding="post", truncating="post"
).tolist()
conversation_df["company_tokens"] = pad_sequences(
    conversation_df["company_tokens"], maxlen=MAX_LEN, padding="post", truncating="post"
).tolist()

print(conversation_df.head())

# Save processed data in chunks to prevent memory overload
# conversation_df.to_csv("data/tokenized_tweets.csv", index=False, chunksize=100000)

sampled_df = conversation_df.sample(n=30000, random_state=42)

#sampled_df.to_csv("data/sampled_tokenized_tweets.csv", index=False)
# df2 = conversation_df.sample(n=3000, random_state=42)  # just 3k pairs
# df2.to_csv("data/small_tokenized_tweets.csv", index=False)

print("Sampled dataset saved.")