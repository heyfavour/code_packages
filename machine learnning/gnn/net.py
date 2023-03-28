import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):  # 1433 16 7
        super().__init__()
        """
        GCNConv
        in_channels：输入通道，比如节点分类中表示每个节点的特征数。
        out_channels：输出通道，最后一层GCNConv的输出通道为节点类别数（节点分类）。
        improved：如果为True表示自环增加，也就是原始邻接矩阵加上2I而不是I，默认为False。
        cached：如果为True，GCNConv在第一次对邻接矩阵进行归一化时会进行缓存，以后将不再重复计算。
        add_self_loops：如果为False不再强制添加自环，默认为True。
        normalize：默认为True，表示对邻接矩阵进行归一化。
        bias：默认添加偏置。
        """
        self.conv1 = GCNConv(in_channels, 256, cached=True, normalize=True)  # 1433 16
        self.conv2 = GCNConv(256, 128, cached=True, normalize=True)  # 16 7
        self.mlp = torch.nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 7),
        )

    def forward(self, x, edge_index, edge_weight=None):
        # x [2708, 1433] edge [2, 10556]
        x = F.dropout(x, p=0.5, training=self.training)  # [2708, 1433]
        x = self.conv1(x, edge_index, edge_weight).relu()  # [2708, 16]
        x = F.dropout(x, p=0.5, training=self.training)  # [2708 16]
        x = self.conv2(x, edge_index, edge_weight)  # [2708 7]
        x = self.mlp(x)
        return x
