import torch
from torch_geometric.data import Data
from src.graphs_classify.models.gin import GINNet

def test_forward():
    edge_index = torch.tensor([[0,1,2,0],[1,2,0,2]], dtype=torch.long)
    x = torch.randn(3, 8)
    data = Data(x=x, edge_index=edge_index, y=torch.tensor([0,0,0]), batch=torch.zeros(3, dtype=torch.long))
    model = GINNet(in_dim=8, hidden=16, out_dim=2)
    out = model(data)
    assert out.shape == (1, 2)
