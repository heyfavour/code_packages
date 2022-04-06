import functools


class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        dp = [1] + [0] * n
        if n == 0: return dp[n]

        dp[1] = dp[0] + 1 if 10 <= int(s[:2]) and 26 else 0
        if n == 1: return dp[n]

        for i in range(2, n):
            if 1 <= int(s[i]) <= 9: dp[i] = dp[i] + dp[i - 1]
            if 10 <= int(s[i - 1:i + 1]) <= 26: dp[i] = dp[i] + dp[i - 2]

        return dp[-1]


if __name__ == '__main__':
    solution = Solution()
    s = "226"
    print(solution.numDecodings(s))
