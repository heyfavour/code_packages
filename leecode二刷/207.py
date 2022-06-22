class Solution:
    def canFinish(self, numCourses: int, prerequisites: list[list[int]]) -> bool:
        #1.构建邻接链表
        #2.判断成环
        pass
if __name__ == '__main__':
    solution = Solution()
    numCourses = 2
    prerequisites = [[1, 0], [0, 1]]
    print(solution.canFinish(numCourses,prerequisites))