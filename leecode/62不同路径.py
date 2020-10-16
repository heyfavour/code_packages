class Solution:
    # 排列组合
    def uniquePaths(self, m: int, n: int) -> int:
        import math
        c = int(math.factorial(m - 1 + n - 1) / (math.factorial(m - 1) * math.factorial(m - 1 + n - 1 - (m - 1))))
        return c
    #动态规划
    def dp_uniquePaths(self, m: int, n: int) -> int:
        dp = [[0 for _ in range(n)] for _ in range(m)]
        dp[0] = [1]*n
        for i in range(m):dp[i][0]=1
        print(dp)
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[i][j]


if __name__ == '__main__':
    s = Solution()
    #print(s.uniquePaths(7, 3))
    print(s.dp_uniquePaths(3,7))
