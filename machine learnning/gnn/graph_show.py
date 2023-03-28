import os.path as osp
import pylab
import networkx as nx

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils.convert import to_networkx


def data_info(data):
    print(data)
    print(data.keys)
    print(data['x'])
    print(data.num_nodes)  # 节点数 2708
    print(data.num_node_features)  # 节点特征数 1433
    print(data.has_self_loops())  # 是否含有孤立点
    print(data.is_directed())  # 是否有向图
    print(data.is_undirected())  # 是否无向图
    print(data.edge_index)  # 邻接关系 双向 两边都要写
    print(data.edge_index.shape)  # [2, 10556]
    L = list(data.edge_index)
    L1 = list(L[0])
    L2 = list(L[1])
    i = L1.index(633)
    print(L1[i-1:i+10])
    print(L2[i-1:i+10])
    print(data.edge_attr)  # 存储边的特征


def nx_draw(data):
    G = to_networkx(data)
    # nx.draw(G, with_labels=True)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edges(G, pos)
    pylab.show()

    # 邻接表
    for line in nx.generate_adjlist(G):
        print(line)


def show_nx():
    G = nx.hexagonal_lattice_graph(2, 3)
    nx.draw(G)
    pylab.show()


def show_plt():
    # settings > tools > Python Scientific
    import matplotlib.pyplot as plt
    price = [100, 250, 380, 500, 700]
    number = [1, 2, 3, 4, 5]
    plt.plot(price, number)
    plt.title("price / number")
    plt.xlabel("price")
    plt.ylabel("number")
    plt.show()


if __name__ == '__main__':
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'Planetoid')
    dataset = Planetoid(path, "Cora", transform=T.NormalizeFeatures())
    data = dataset[0]
    data_info(data)
    # nx_draw(data)
