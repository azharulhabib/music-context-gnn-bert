import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class GATTagClassifier(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=128, n_tags=10, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.head = nn.Linear(hidden_dim, n_tags)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        g = global_mean_pool(x, batch)
        return self.head(g)