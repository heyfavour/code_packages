from collections import defaultdict, deque
class Solution:
    def canFinish(self, numCourses: int, prerequisites: list[list[int]]) -> bool:
        # 1.构建邻接链表

        graph = defaultdict(list)
        in_degree = [0]*numCourses
        for course in prerequisites:
            graph[course[1]].append(course[0])
            in_degree[course[0]] = in_degree[course[0]] + 1

        # 2.判断成环 入度 BFS 出度 DFS
        # 入度为0弹出 其指向的节点入度减1
        Q = deque()
        visted= []
        for i in range(numCourses):
            if in_degree[i] == 0:
                Q.append(i)
                visted.append(i)
        while Q:
            for i in range(len(Q)):
                node = Q.popleft()
                for out in graph[node]:
                    in_degree[out] = in_degree[out] - 1
                    if in_degree[out] == 0:
                        Q.append(out)
                        visted.append(out)
        return len(visted) == numCourses






if __name__ == '__main__':
    solution = Solution()
    numCourses = 2
    prerequisites = [[1, 0], [0, 1]]
    print(solution.canFinish(numCourses, prerequisites))
