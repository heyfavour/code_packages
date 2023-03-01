"""
按边权升序排序；//重新编号
T=空集；
FOR j = 1 TO m
    IF T U [j] 不成环
        将j加入T;
"""
import heapq
class UF:
    def __init__(self, n=0):
        self.count = n
        self.parent = list(range(n+1))

    def find(self, x):
        if (x == self.parent[x]):return x
        return self.find(self.parent[x])

    def connect(self, x, y):
        x_par = self.find(x)
        y_par = self.find(y)
        if (x_par != y_par):self.parent[y_par] = x_par

def kruskal(matrix):
    n = len(matrix)
    uf = UF(n)

    stack = []
    edges = []
    for i in range(n):
        for j in range(n):
            edge = matrix[i][j]
            if edge!=-1:heapq.heappush(stack,(edge,(i,j,edge)))
    while stack:
        edge,(i,j,edge) = heapq.heappop(stack)
        print(i,j,edge)
        if uf.find(i)!=uf.find(j):
            uf.connect(i,j)
            edges.append(edge)
    print(sum(edges))



if __name__ == '__main__':
    city = [
            [-1, 8, 7, -1, -1, -1, -1],
            [8, -1, 6, -1, 9, 8, -1],
            [7, 6, -1, 3, 4, -1, -1],
            [-1, -1, 3, -1, 2, -1, -1],
            [-1, 9, 4, 2, -1, -1, 10],
            [-1, 8, -1, -1, -1, -1, 2],
            [-1, -1, -1, -1, 10, 2, -1]
    ]
    kruskal(city)
