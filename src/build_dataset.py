import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import os

from audio_features import process_track
from graph_builder import build_segment_graph, graph_to_arrays


def track_to_pyg_graph(mp3_path):
    _, chroma_segs = process_track(mp3_path)
    if len(chroma_segs) < 2:
        return None

    G = build_segment_graph(chroma_segs)
    node_features, edge_index, edge_weights = graph_to_arrays(G)

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_mini_dataset(n_samples=10):
    annotations = pd.read_csv("data/raw/magnatagatune_annotations.csv", sep="\t")
    audio_dir = "data/raw/magnatagatune_audio"

    graphs = []
    labels = []
    tag_columns = [c for c in annotations.columns if c not in ("clip_id", "mp3_path")]

    count = 0
    for _, row in annotations.iterrows():
        if count >= n_samples:
            break

        mp3_path = os.path.join(audio_dir, row["mp3_path"])
        if not os.path.exists(mp3_path):
            continue

        graph = track_to_pyg_graph(mp3_path)
        if graph is None:
            continue

        label = torch.tensor(row[tag_columns].values.astype(np.float32))
        graphs.append(graph)
        labels.append(label)
        count += 1
        print(f"Processed {count}/{n_samples}: {row['mp3_path']}")

    return graphs, labels


if __name__ == "__main__":
    graphs, labels = build_mini_dataset(n_samples=10)
    print(f"\nBuilt {len(graphs)} graphs")
    print(f"Sample graph: {graphs[0]}")
    print(f"Sample label shape: {labels[0].shape}")
    print(f"Number of tags: {labels[0].shape[0]}")