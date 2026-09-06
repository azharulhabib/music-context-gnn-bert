import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_LEN = 128
TOP_K_TAGS = 50
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5


def load_musiccaps(path="data/raw/musiccaps.csv"):
    df = pd.read_csv(path)
    df["aspects"] = df["aspect_list"].apply(ast.literal_eval)
    return df


def build_tag_vocab(df, top_k=TOP_K_TAGS):
    all_tags = [tag for aspects in df["aspects"] for tag in aspects]
    counts = pd.Series(all_tags).value_counts()
    return counts.head(top_k).index.tolist()


def build_labels(df, vocab):
    labels = np.zeros((len(df), len(vocab)), dtype=np.float32)
    for i, aspects in enumerate(df["aspects"]):
        for tag in aspects:
            if tag in vocab:
                labels[i, vocab.index(tag)] = 1.0
    return labels


class CaptionTagDataset(Dataset):
    def __init__(self, captions, labels, tokenizer, max_len=MAX_LEN):
        self.captions = captions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        tokens = self.tokenizer(
            self.captions[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in tokens.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class BertTagClassifier(nn.Module):
    def __init__(self, n_tags, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.head = nn.Linear(768, n_tags)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > threshold).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    return total_loss / len(loader), macro_f1, micro_f1


def main():
    df = load_musiccaps()
    vocab = build_tag_vocab(df)
    labels = build_labels(df, vocab)
    captions = df["caption"].tolist()

    train_captions, val_captions, train_labels, val_labels = train_test_split(
        captions, labels, test_size=0.2, random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_ds = CaptionTagDataset(train_captions, train_labels, tokenizer)
    val_ds = CaptionTagDataset(val_captions, val_labels, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = BertTagClassifier(n_tags=len(vocab)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, macro_f1, micro_f1 = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Macro-F1: {macro_f1:.4f} | Micro-F1: {micro_f1:.4f}")

    torch.save(model.state_dict(), "results/bert_tag_classifier.pt")
    print("Model saved to results/bert_tag_classifier.pt")


if __name__ == "__main__":
    main()