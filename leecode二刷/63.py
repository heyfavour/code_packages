class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: list[list[int]]) -> int:
        dp = [[0] * len(obstacleGrid[0]) for i in range(len(obstacleGrid))]
        for i in range(len(obstacleGrid[0])):
            if obstacleGrid[0][i] == 1: break
            dp[0][i] = 1
        for i in range(len(obstacleGrid)):
            if obstacleGrid[i][0] == 1: break
            dp[i][0] = 1

        for i in range(1, len(obstacleGrid)):
            for j in range(1, len(obstacleGrid[0])):
                if obstacleGrid[i][j] == 1: continue
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]


if __name__ == '__main__':
    obstacleGrid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    solution = Solution()
    print(solution.uniquePathsWithObstacles(obstacleGrid))
