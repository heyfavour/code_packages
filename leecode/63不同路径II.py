from typing import List


class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(n):
            if obstacleGrid[0][i] == 1:break
            dp[0][i] = 1
        for i in range(m):
            if obstacleGrid[i][0] == 1:break
            dp[i][0] = 1
        for i in range(1,m):
            for j in range(1,n):
                if obstacleGrid[i][j]==1:
                    dp[i][j] = 0
                else:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return  dp[i][j]


if __name__ == '__main__':
    L = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    L = [[1,0],[0,0]]
    L = [[0,0],[1,1],[0,0]]
    s = Solution()
    print((s.uniquePathsWithObstacles(L)))
