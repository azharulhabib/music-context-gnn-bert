import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score

from audio_features import process_track
from graph_builder import build_segment_graph, graph_to_arrays

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_SAMPLES = 2000
GENRE_TAGS = ["classical", "rock", "jazz", "electronic", "pop", "ambient",
              "metal", "folk", "country", "techno"]
BATCH_SIZE = 8
EPOCHS = 10
LR = 2e-5
HIDDEN_DIM = 128
MAX_LEN = 64


def track_to_graph(mp3_path):
    mel_segs, _ = process_track(mp3_path)
    if len(mel_segs) < 2:
        return None
    G = build_segment_graph(mel_segs)
    node_features, edge_index, edge_weights = graph_to_arrays(G)
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def build_pseudo_caption(row, non_genre_tags):
    present = [t for t in non_genre_tags if row[t] == 1]
    if not present:
        return "unknown sound"
    return " ".join(present)


def build_split_dataset(annotations, audio_dir, non_genre_tags, clip_ids, n_samples):
    subset = annotations[annotations["clip_id"].isin(clip_ids)].reset_index(drop=True)

    graphs, captions = [], []
    count = 0
    for _, row in subset.iterrows():
        if count >= n_samples:
            break
        mp3_path = os.path.join(audio_dir, row["mp3_path"])
        if not os.path.exists(mp3_path):
            continue
        graph = track_to_graph(mp3_path)
        if graph is None:
            continue
        caption = build_pseudo_caption(row, non_genre_tags)
        label = torch.tensor(row[GENRE_TAGS].values.astype(np.float32))
        graph.y = label.unsqueeze(0)
        graphs.append(graph)
        captions.append(caption)
        count += 1
        if count % 100 == 0:
            print(f"Built {count}/{n_samples} samples")
    return graphs, captions

def build_dataset(n_samples=N_SAMPLES):
    annotations = pd.read_csv("data/raw/magnatagatune_annotations.csv", sep="\t")
    audio_dir = "data/raw/magnatagatune_audio"

    all_tag_cols = [c for c in annotations.columns if c not in ("clip_id", "mp3_path")]
    non_genre_tags = [t for t in all_tag_cols if t not in GENRE_TAGS]

    mask = annotations[GENRE_TAGS].sum(axis=1) > 0
    annotations = annotations[mask].reset_index(drop=True)

    train_ids = set(pd.read_csv("data/splits/train_ids.csv")["clip_id"])
    val_ids = set(pd.read_csv("data/splits/val_ids.csv")["clip_id"])

    n_train = int(n_samples * 0.8)
    n_val = n_samples - n_train

    print("Building train samples...")
    train_graphs, train_captions = build_split_dataset(annotations, audio_dir, non_genre_tags, train_ids, n_train)
    print("Building val samples...")
    val_graphs, val_captions = build_split_dataset(annotations, audio_dir, non_genre_tags, val_ids, n_val)

    return train_graphs, train_captions, val_graphs, val_captions


class FusionModel(nn.Module):
    def __init__(self, gnn_in_dim=128, gnn_hidden=HIDDEN_DIM, bert_dim=768, n_tags=len(GENRE_TAGS)):
        super().__init__()
        self.conv1 = SAGEConv(gnn_in_dim, gnn_hidden)
        self.conv2 = SAGEConv(gnn_hidden, gnn_hidden)

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.q_proj = nn.Linear(gnn_hidden, gnn_hidden)
        self.k_proj = nn.Linear(bert_dim, gnn_hidden)
        self.v_proj = nn.Linear(bert_dim, gnn_hidden)

        self.head = nn.Linear(gnn_hidden * 2, n_tags)

    def encode_graph(self, x, edge_index, batch):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        return global_mean_pool(h, batch)

    def forward(self, x, edge_index, batch, input_ids, attention_mask):
        g = self.encode_graph(x, edge_index, batch)

        input_ids = input_ids.view(-1, MAX_LEN)
        attention_mask = attention_mask.view(-1, MAX_LEN)

        H = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        Q = self.q_proj(g).unsqueeze(1)
        K = self.k_proj(H)
        V = self.v_proj(H)

        attn = torch.softmax(Q @ K.transpose(-2, -1) / (K.size(-1) ** 0.5), dim=-1)
        attended = (attn @ V).squeeze(1)

        z = torch.cat([g, attended], dim=-1)
        return self.head(z)


class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, captions, tokenizer, max_len=MAX_LEN):
        self.graphs = graphs
        self.captions = captions
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        tokens = self.tokenizer(
            self.captions[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        graph = self.graphs[idx]
        graph.input_ids = tokens["input_ids"].squeeze(0)
        graph.attention_mask = tokens["attention_mask"].squeeze(0)
        return graph


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch, batch.input_ids, batch.attention_mask)
        loss = criterion(out, batch.y)
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
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.batch, batch.input_ids, batch.attention_mask)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            preds = (torch.sigmoid(out) > threshold).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    return total_loss / len(loader), macro_f1, micro_f1


def main():
    print("Building dataset...")
    train_graphs, train_captions, val_graphs, val_captions = build_dataset()
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_ds = FusionDataset(train_graphs, train_captions, tokenizer)
    val_ds = FusionDataset(val_graphs, val_captions, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = FusionModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_macro_f1 = 0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, macro_f1, micro_f1 = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Macro-F1: {macro_f1:.4f} | Micro-F1: {micro_f1:.4f}")
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), "results/fusion_best.pt")
    print(f"Best Macro-F1: {best_macro_f1:.4f}")
    print("Best model saved to results/fusion_best.pt")



if __name__ == "__main__":
    main()