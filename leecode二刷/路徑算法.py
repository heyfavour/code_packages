import pprint


def floyd():
    nodes = ('A', 'B', 'C', 'D', 'E')  # 节点
    # dis矩阵初始化
    inf = float("inf")
    dis = [[0, 3, 4, inf, 1],
           [3, 0, inf, 5, 1],
           [4, inf, 0, 2, 2],
           [inf, 5, 2, 0, inf],
           [1, 1, 2, inf, 0]]
    pprint.pprint(dis)
    node_num = len(nodes)  # 节点个数
    # floyd算法 map[i,j]:=min{map[i,k]+map[k,j],map[i,j]},map[i,j]表示i到j的最短距离，K是穷举i,j的断点，
    for k in range(node_num):
        for i in range(node_num):
            for j in range(node_num):
                # dis[i][j]表示i到j的最短距离，dis[i][k]表示i到k的最短距离，dis[k][j]表示k到j的最短距离
                dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j])  # 若比原来距离小，则更新
    print('各个点的最短路径为:')
    pprint.pprint(dis)

if __name__ == '__main__':
    floyd()
