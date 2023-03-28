"""
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
"""
import torch
import os.path as osp
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from net import GCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_wandb(name=f'GCN-Cora', lr=0.01, epochs=200, hidden_channels=16, device="cpu")



def train():
    model.train()
    optimizer.zero_grad()
    # data.x [2708, 1433])
    # data.edge_index [2, 10556]
    # data.edge_attr None
    out = model(data.x, data.edge_index, data.edge_attr)#[2708 7]
    #data.train_mask [2708] BoolTensor out[data.train_mask] -> [140 7] 切片
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])#[140 7] [140]
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)#[2408,7]->[2408 1]
    accs = []

    for mask in [data.train_mask, data.val_mask, data.test_mask]:#140 500 1000
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


if __name__ == '__main__':
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'Planetoid')
    dataset = Planetoid(path, "Cora", transform=T.NormalizeFeatures())
    data = dataset[0]
    """
    x: 用于存储每个节点的特征，形状是[num_nodes, num_node_features]。
    edge_index: 用于存储节点之间的边，形状是 [2, num_edges]。
    pos: 存储节点的坐标，形状是[num_nodes, num_dimensions]。
    y: 存储样本标签。如果是每个节点都有标签，那么形状是[num_nodes, *]；如果是整张图只有一个标签，那么形状是[1, *]。
    edge_attr: 存储边的特征。形状是[num_edges, num_edge_features]。 #按照 edge_index 顺序排列
    """
    model = GCN(1433, 16, 7)
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.AdamW([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0),
        dict(params=model.mlp.parameters()),
    ], lr=0.01)  # Only perform weight-decay on first convolution.

    for epoch in range(1, 500 + 1):
        loss = train()
        train_acc, val_acc, test_acc = test()
        log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
