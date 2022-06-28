class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        # 构建map
        from collections import defaultdict,deque
        graph = defaultdict(list)
        indegree = [0]*numCourses
        for link in prerequisites:
            graph[link[1]].append(link[0])
            indegree[link[0]] = indegree[link[0]] + 1
        Q = deque()
        result = []
        for i in range(numCourses):
            if indegree[i]==0:
                Q.append(i)
                result.append(i)
        while Q:
            for i in range(len(Q)):
                node = Q.popleft()
                for i in graph[node]:
                    indegree[i] = indegree[i]-1
                    if indegree[i]==0:
                        Q.append(i)
                        result.append(i)
        return result if len(result) == numCourses else []

if __name__ == '__main__':
    solution = Solution()
    numCourses = 4
    prerequisites = [[1, 0], [2, 0], [3, 1], [3, 2]]
    print(solution.findOrder(numCourses, prerequisites))
