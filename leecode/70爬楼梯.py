class Solution:
    @classmethod
    def climbStairs(self, n: int) -> int:
        dp = [0] * n
        for i in range(n):
            if i <= 2:
                dp[i] = i + 1
            else:
                dp[i] = dp[i - 1] + dp[i - 2]
        return dp[-1]


if __name__ == '__main__':
    print(Solution.climbStairs(3))
