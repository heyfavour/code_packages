import collections


class Solution(object):
    def calcEquation(self, equations, values, queries):
        # graph = collections.defaultdict(dict)
        # for (a, b), v in zip(equations, values):
        #     graph[a][b] = v
        #     graph[b][a] = 1 / v
        # for k in graph.keys():
        #     for i in graph.keys():
        #         for j in graph.keys():
        #             if k in graph[i].keys() and j in graph[k].keys() and j not in graph[i].keys():
        #                 graph[i][j] = graph[i][k] * graph[k][j]
        #
        # res = []
        # for a, b in queries:
        #     if b in graph[a]:
        #         res.append(graph[a][b])
        #     else:
        #         res.append(-1)
        # return res
        nodes = set()
        for a, b in equations:
            nodes.add(a)
            nodes.add(b)
        nodes = sorted(list(nodes))
        n = len(nodes)
        graph = [[float("inf")] * n for _ in range(n)]
        for i in range(n):graph[i][i] = 1
        nodes_dict = {v: k for k, v in enumerate(nodes)}
        for (a, b), v in zip(equations, values):
            graph[nodes_dict[a]][nodes_dict[b]] = v
            graph[nodes_dict[b]][nodes_dict[a]] = 1 / v
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    graph[i][j] = min(graph[i][j], graph[i][k] * graph[k][j])
        print(graph)
        ans = []
        for a, b in queries:
            if  nodes_dict.get(a) is None or  nodes_dict.get(b) is None:
                ans.append(-1)
            else:
                route = graph[nodes_dict[a]][nodes_dict[b]]
                if route == float("inf"):
                    ans.append(-1)
                else:
                    ans.append(route)
        return ans


if __name__ == '__main__':
    equations = [["a", "b"], ["b", "c"]]
    values = [2.0, 3.0]
    queries = [["a", "c"], ["b", "a"], ["a", "e"]]
    solution = Solution()
    print(solution.calcEquation(equations, values, queries))
