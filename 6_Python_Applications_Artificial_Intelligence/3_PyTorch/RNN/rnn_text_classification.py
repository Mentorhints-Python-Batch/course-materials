"""
RNN-based Multi-Class Text Classification on Women's Clothing E-Commerce Reviews
=================================================================================
Predicts the Department Name (Tops, Dresses, Bottoms, Intimate, Jackets, Trend)
from the review text using a Bidirectional LSTM with attention.

Dataset: Womens Clothing E-Commerce Reviews.csv
Target: Department Name (6 classes)
"""

import os
import re
import time
import string
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = os.path.join(os.path.dirname(__file__), "Dataset", "Womens Clothing E-Commerce Reviews.csv")

# ---------------------------------------------------------------------------
# 1. Text Preprocessing & Vocabulary (shared utilities)
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class Vocabulary:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, max_size: int = 25_000, min_freq: int = 2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.token2idx: dict[str, int] = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2token: dict[int, str] = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

    def build(self, texts: list[str]):
        counter = Counter()
        for t in texts:
            counter.update(t.split())
        idx = len(self.token2idx)
        for word, freq in counter.most_common(self.max_size):
            if freq < self.min_freq:
                continue
            if word not in self.token2idx:
                self.token2idx[word] = idx
                self.idx2token[idx] = word
                idx += 1

    def encode(self, text: str, max_len: int) -> list[int]:
        tokens = text.split()[:max_len]
        indices = [self.token2idx.get(t, 1) for t in tokens]
        return indices + [0] * (max_len - len(indices))

    def __len__(self):
        return len(self.token2idx)


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.vocab.encode(self.texts[idx], self.max_len)
        return (
            torch.tensor(enc, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------------------

class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM with max-pooling over timesteps."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 n_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, _ = self.lstm(embedded)              # (B, T, 2*H)
        pooled, _ = output.max(dim=1)                # max-pool over time
        return self.classifier(pooled)


class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM with self-attention for multi-class classification."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 n_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()        # (B, T, 1)
        embedded = self.dropout(self.embedding(x))
        output, _ = self.lstm(embedded)               # (B, T, 2*H)

        attn_scores = self.attention(output)          # (B, T, 1)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = (output * attn_weights).sum(dim=1)  # (B, 2*H)

        return self.classifier(context)


class BiGRUClassifier(nn.Module):
    """Bidirectional GRU with average pooling."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 n_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
                          batch_first=True, dropout=dropout, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()
        embedded = self.dropout(self.embedding(x))
        output, _ = self.gru(embedded)
        lengths = mask.sum(dim=1).clamp(min=1)
        avg_pool = (output * mask).sum(dim=1) / lengths
        return self.classifier(avg_pool)


# ---------------------------------------------------------------------------
# 4. Training & Evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for texts, labels in loader:
        texts, labels = texts.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(texts)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for texts, labels in loader:
        texts, labels = texts.to(DEVICE), labels.to(DEVICE)
        logits = model(texts)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += len(labels)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# ---------------------------------------------------------------------------
# 5. Data Loading
# ---------------------------------------------------------------------------

def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0)
    df = df.dropna(subset=["Review Text", "Department Name"])
    df["clean_text"] = df["Review Text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["Department Name"])

    texts = df["clean_text"].tolist()
    labels = df["label"].tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    vocab = Vocabulary(max_size=25_000, min_freq=2)
    vocab.build(X_train)

    return X_train, X_val, X_test, y_train, y_val, y_test, vocab, le


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    print(f"Using device: {DEVICE}\n")

    X_train, X_val, X_test, y_train, y_val, y_test, vocab, label_encoder = load_data()
    num_classes = len(label_encoder.classes_)
    class_names = label_encoder.classes_.tolist()

    print(f"Vocabulary size : {len(vocab):,}")
    print(f"Number of classes: {num_classes} — {class_names}")
    print(f"Train / Val / Test : {len(X_train):,} / {len(X_val):,} / {len(X_test):,}")

    MAX_LEN = 200
    BATCH_SIZE = 64
    EMBED_DIM = 128
    HIDDEN_DIM = 128
    N_LAYERS = 2
    DROPOUT = 0.3
    LR = 1e-3
    EPOCHS = 5

    train_ds = ReviewDataset(X_train, y_train, vocab, MAX_LEN)
    val_ds = ReviewDataset(X_val, y_val, vocab, MAX_LEN)
    test_ds = ReviewDataset(X_test, y_test, vocab, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    models_config = {
        "BiLSTM (MaxPool)": BiLSTMClassifier,
        "BiLSTM (Attention)": BiLSTMAttention,
        "BiGRU (AvgPool)": BiGRUClassifier,
    }

    for name, ModelClass in models_config.items():
        print(f"\n{'='*60}")
        print(f"  Training {name}")
        print(f"{'='*60}")

        model = ModelClass(
            vocab_size=len(vocab),
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_classes=num_classes,
            n_layers=N_LAYERS,
            dropout=DROPOUT,
        ).to(DEVICE)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total_params:,}\n")

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        best_val_loss = float("inf")
        save_path = os.path.join(
            os.path.dirname(__file__),
            f"best_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pt",
        )

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
            elapsed = time.time() - t0

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

        model.load_state_dict(torch.load(save_path, weights_only=True))
        test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion)

        print(f"\n{name} Test Results  —  Loss: {test_loss:.4f}  Acc: {test_acc:.4f}")
        print(classification_report(labels, preds, target_names=class_names))


if __name__ == "__main__":
    main()
