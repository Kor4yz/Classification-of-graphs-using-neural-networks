import torch
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool

class GINNet(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=5, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.batch = nn.ModuleList()
        dim_prev = in_dim if in_dim > 0 else 1
        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(dim_prev, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINConv(mlp))
            self.batch.append(nn.BatchNorm1d(hidden))
            dim_prev = hidden
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if x is None:  # некоторые TUDataset без признаков
            x = torch.ones((data.num_nodes, 1), device=edge_index.device)
        for conv, bn in zip(self.convs, self.batch):
            x = conv(x, edge_index)
            x = bn(x).relu()
        x = global_add_pool(x, batch)
        return self.head(x)
