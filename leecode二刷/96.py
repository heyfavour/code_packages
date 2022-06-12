class Solution:
    def numTrees(self, n: int) -> int:
        # G(N)= F(1)+F(2)+F(3)+ ... +F(N)
        # G(N) = G(0)G(n-1)+G(1)G(N-2)+G(2)G(N-2)...G(N-1)*G(0)
        # G(0) = 1 G(1) = 1
        dp = [0] * (n + 1)
        dp[0] = dp[1] = 1
        for i in range(2, n + 1):
            for j in range(i):
                dp[i] = dp[i] + dp[j] * dp[i - j - 1]
        return dp[-1]


if __name__ == '__main__':
    solution = Solution()
    print(solution.numTrees(5))
