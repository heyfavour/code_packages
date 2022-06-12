class Solution:
    def minimumTotal(self, triangle: list[list[int]]) -> int:
        # 1
        # 1 1
        # 1 2 1
        n = len(triangle)
        dp = [[0] * (i + 1) for i in range(n)]
        for i in range(n):
            for j in range(i + 1):
                if j == 0:
                    dp[i][j] = dp[i - 1][j] + triangle[i][j]
                elif j == i:
                    dp[i][j] = dp[i - 1][j - 1] + triangle[i][j]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle[i][j]
        return min(dp[-1])


if __name__ == '__main__':
    triangle = [[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]]
    solution = Solution()
    print(solution.minimumTotal(triangle))
