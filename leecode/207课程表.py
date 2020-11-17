from typing import List
import collections


class Solution:
    @classmethod
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        edges = collections.defaultdict(list)
        indeg = [0] * numCourses

        for info in prerequisites:
            edges.setdefault(info[1], []).append(info[0])  # 临接链表
            indeg[info[0]] = indeg[info[0]] + 1  # 多少次被指向
        q = collections.deque([u for u in range(numCourses) if indeg[u] == 0])  # 记录出发点
        visited = 0
        while q:
            print(q)
            visited = visited + 1
            u = q.popleft()
            for v in edges[u]:
                indeg[v] = indeg[v] - 1
                if indeg[v] == 0: q.append(v)
        return visited == numCourses


if __name__ == '__main__':
    numCourses, prerequisites = 2, [[1, 0], [0, 1]]
    print(Solution.canFinish(numCourses, prerequisites))
