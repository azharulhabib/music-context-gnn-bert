import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool


class GNNTagClassifier(nn.Module):
    def __init__(self, in_dim=12, hidden_dim=64, n_tags=188):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, n_tags)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        g = global_mean_pool(x, batch)
        return self.head(g)


if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    from build_dataset import build_mini_dataset

    graphs, labels = build_mini_dataset(n_samples=10)
    for g, l in zip(graphs, labels):
        g.y = l.unsqueeze(0)

    loader = DataLoader(graphs, batch_size=4, shuffle=True)
    model = GNNTagClassifier()

    batch = next(iter(loader))
    out = model(batch.x, batch.edge_index, batch.batch)
    print(f"Batch size: {batch.num_graphs}")
    print(f"Output shape: {out.shape}")
    print(f"Target shape: {batch.y.shape}")