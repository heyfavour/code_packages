from typing import List
import numpy


class Solution:
    @classmethod
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    dp[0][0] =grid[i][j]
                elif i == 0 and j>0:
                    dp[i][j] = dp[i][j-1]  + grid[i][j]
                elif j == 0 and i>0:
                    dp[i][j] = dp[i-1][j]  + grid[i][j]
                else:
                    dp[i][j] = min(dp[i-1][j],dp[i][j-1]) + grid[i][j]
        return dp[i][j]


if __name__ == '__main__':
    L = [
        [1, 3, 1],
        [1, 5, 1],
        [4, 2, 1]
    ]

    print(Solution.minPathSum(L))
