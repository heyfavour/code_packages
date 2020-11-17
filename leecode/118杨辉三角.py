from typing import List


class Solution:
    @classmethod
    def generate(self, numRows: int) -> List[List[int]]:
        dp = [[1] * (i + 1) for i in range(numRows)]
        for i in range(numRows):
            if i in (0, 1):
                continue
            else:
                for j in range(1, len(dp[i])-1):
                    dp[i][j] = dp[i - 1][j - 1] + dp[i-1][j]
        return dp


if __name__ == '__main__':
    print(Solution.generate(5))
