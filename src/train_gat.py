import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from audio_features import process_track
from graph_builder import build_segment_graph, graph_to_arrays
from gat_model import GATTagClassifier

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_SAMPLES = 2000
GENRE_TAGS = ["classical", "rock", "jazz", "electronic", "pop", "ambient",
              "metal", "folk", "country", "techno"]
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-3


def track_to_pyg_graph(mp3_path):
    mel_segs, _ = process_track(mp3_path)
    if len(mel_segs) < 2:
        return None
    G = build_segment_graph(mel_segs)
    node_features, edge_index, edge_weights = graph_to_arrays(G)
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_split_dataset(annotations, audio_dir, clip_ids, n_samples):
    subset = annotations[annotations["clip_id"].isin(clip_ids)].reset_index(drop=True)
    graphs = []
    count = 0
    for _, row in subset.iterrows():
        if count >= n_samples:
            break
        mp3_path = os.path.join(audio_dir, row["mp3_path"])
        if not os.path.exists(mp3_path):
            continue
        graph = track_to_pyg_graph(mp3_path)
        if graph is None:
            continue
        label = torch.tensor(row[GENRE_TAGS].values.astype(np.float32))
        graph.y = label.unsqueeze(0)
        graphs.append(graph)
        count += 1
        if count % 100 == 0:
            print(f"Built {count}/{n_samples} graphs")
    return graphs


def build_dataset(n_samples=N_SAMPLES):
    annotations = pd.read_csv("data/raw/magnatagatune_annotations.csv", sep="\t")
    audio_dir = "data/raw/magnatagatune_audio"

    mask = annotations[GENRE_TAGS].sum(axis=1) > 0
    annotations = annotations[mask].reset_index(drop=True)

    train_ids = set(pd.read_csv("data/splits/train_ids.csv")["clip_id"])
    val_ids = set(pd.read_csv("data/splits/val_ids.csv")["clip_id"])

    n_train = int(n_samples * 0.8)
    n_val = n_samples - n_train

    print("Building train graphs...")
    train_graphs = build_split_dataset(annotations, audio_dir, train_ids, n_train)
    print("Building val graphs...")
    val_graphs = build_split_dataset(annotations, audio_dir, val_ids, n_val)

    return train_graphs, val_graphs


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
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
            out = model(batch.x, batch.edge_index, batch.batch)
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
    train_graphs, val_graphs = build_dataset()
    print(f"Train graphs: {len(train_graphs)}, Val graphs: {len(val_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE)

    model = GATTagClassifier(n_tags=len(GENRE_TAGS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_macro_f1 = 0
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, macro_f1, micro_f1 = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Macro-F1: {macro_f1:.4f} | Micro-F1: {micro_f1:.4f}")
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), "results/gat_best.pt")

    print(f"Best Macro-F1: {best_macro_f1:.4f}")
    print("Best model saved to results/gat_best.pt")


if __name__ == "__main__":
    main()