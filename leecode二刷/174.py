class Solution:
    def calculateMinimumHP(self, dungeon: list[list[int]]) -> int:
        m, n = len(dungeon), len(dungeon[0])
        dp = [[float("inf")] * n for _ in range(m)]
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if i == m - 1 and j == n - 1:
                    dp[i][j] = max(-dungeon[i][j], 0)
                elif i == m - 1 and j < n - 1:
                    dp[i][j] = max(dp[i][j + 1] - dungeon[i][j], 0)
                elif j == n - 1 and i < m - 1:
                    dp[i][j] = max(dp[i + 1][j] - dungeon[i][j], 0)
                else:
                    dp[i][j] = max(min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j], 0)
        return dp[0][0] + 1


if __name__ == '__main__':
    solution = Solution()
    dungeon = [[-2, -3, 3], [-5, -10, 1], [10, 30, -5]]
    dungeon = [[100]]
    dungeon = [[1, -3, 3], [0, -2, 0], [-3, -3, -3]]
    print(solution.calculateMinimumHP(dungeon))
