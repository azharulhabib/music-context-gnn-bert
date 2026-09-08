import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_THRESHOLD = 0.8


def pool_segment(segment):
    return segment.mean(axis=1)


def build_segment_graph(feature_segments, tau=SIMILARITY_THRESHOLD):
    n = len(feature_segments)
    G = nx.Graph()

    pooled = [pool_segment(seg) for seg in feature_segments]

    for i in range(n):
        G.add_node(i, feat=pooled[i])

    for i in range(n - 1):
        G.add_edge(i, i + 1, weight=1.0, edge_type="temporal")

    sims = cosine_similarity(pooled)
    for i in range(n):
        for j in range(i + 2, n):
            if sims[i, j] > tau:
                G.add_edge(i, j, weight=float(sims[i, j]), edge_type="similarity")

    return G


def graph_to_arrays(G):
    node_features = np.array([G.nodes[i]["feat"] for i in G.nodes])
    edges = list(G.edges())
    edge_index = np.array(edges).T if edges else np.empty((2, 0), dtype=int)
    edge_weights = np.array([G.edges[e]["weight"] for e in edges])
    return node_features, edge_index, edge_weights


if __name__ == "__main__":
    import pandas as pd
    import os
    from audio_features import process_track

    annotations = pd.read_csv("data/raw/magnatagatune_annotations.csv", sep="\t")
    sample_path = annotations.iloc[0]["mp3_path"]
    full_path = os.path.join("data/raw/magnatagatune_audio", sample_path)

    print(f"Building graph for: {full_path}")
    mel_segs, chroma_segs = process_track(full_path)

    G = build_segment_graph(chroma_segs)

    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Edge types: {[G.edges[e]['edge_type'] for e in G.edges()]}")

    node_features, edge_index, edge_weights = graph_to_arrays(G)
    print(f"Node features shape: {node_features.shape}")
    print(f"Edge index shape: {edge_index.shape}")