from typing import List
import  collections
class Solution:
    @classmethod
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        grap_dict = collections.defaultdict(list)
        count = [0]*numCourses
        for i in prerequisites:
            grap_dict[i[1]].append(i[0])#邻接链表
            count[i[0]] = count[i[0]] + 1#有向表
        q = collections.deque([i for i in range(numCourses) if count[i]==0])#可触发的点
        visit = 0
        result = []
        while q:
            u = q.popleft()
            visit = visit + 1
            result.append(u)
            for i in grap_dict[u]:
                count[i] = count[i] - 1
                if count[i] == 0:q.append(i)

        if visit == numCourses:return result
        return []



if __name__ == '__main__':
     numCourses,prerequisites= 4, [[1, 0], [2, 0], [3, 1], [3, 2]]
     print(Solution.findOrder(numCourses,prerequisites))
