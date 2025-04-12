# Mini LLM Transformer

## Overview

The **Mini LLM Transformer** is a transformer-based model designed for sequence-to-sequence tasks like text generation and completion. The model architecture is based on the paper "*Attention is All You Need*" and utilizes **ELECTRA** embeddings for processing input data. This project combines conversational data with a **Twitter tweets dataset** to fine-tune a model capable of generating meaningful and coherent responses.


## Features

- **Transformer-based Seq2Seq Model**: Implements the transformer architecture for sequence-to-sequence tasks, capable of text generation and completion.
- **Pretrained Embeddings**: Uses **ELECTRA** embeddings for input text, which are fine-tuned on custom datasets.
- **Beam Search Decoding**: Incorporates beam search decoding to generate more accurate and diverse sequences.
- **Multi-Head Attention**: The model uses multi-head attention to process sequences in parallel and capture long-range dependencies.
- **Training and Evaluation**: The repository includes scripts for training the model on custom datasets and evaluating its performance.


## Dataset

The datasets used for training the model include:

1. **Conversational Data**: A dataset containing pairs of **questions** and **answers** in a conversational format.
   - Format: CSV file with `input` (questions) and `output` (answers).

2. **Twitter Tweets Dataset**: A dataset of **real-world tweets** used to fine-tune the model on social media-style text. 
   - Format: CSV file with `text` (tweets).

**Note**: Due to the size of the datasets, they are not included in this repository. You can download them from the following [Google Drive link](https://drive.google.com/drive/folders/1CvslonMY5aiB4GS6U7_3QtoEuCTq3pAl?usp=sharing).


## Model Architecture

The **Mini LLM Transformer** follows the architecture outlined in "*Attention is All You Need*". Here's a breakdown of the model:

- **ELECTRA Embeddings**: The model utilizes **ELECTRA** pre-trained embeddings for input representation, which are fine-tuned on specific datasets.
- **Transformer Encoder**: The encoder uses multi-head attention mechanisms to learn relationships between tokens in the input sequence.
- **Transformer Decoder**: The decoder generates output sequences based on the encoder's memory and previously generated tokens.
- **Positional Encoding**: Since transformers do not inherently understand the order of tokens, **positional encodings** are added to provide sequence order information.
- **Final Output Layer**: The output from the decoder is passed through a final fully connected layer to produce a probability distribution over the vocabulary.


## Training

### Training Configuration

- **EPOCHS**: Number of epochs for training.
- **BATCH_SIZE**: Size of the mini-batch.
- **NHEAD**: Number of attention heads.
- **DMODEL**: Dimension of the model (embedding size).
- **NUM_LAYERS**: Number of transformer layers.
- **MAX_LEN**: Maximum sequence length.
- **LEARNING_RATE**: Learning rate for AdamW optimizer.

Feel free to **clone this repository**, make improvements, or try out new ideas to **enhance the model**. If you have better ways to optimize or extend this model, your contributions are most welcome. 
