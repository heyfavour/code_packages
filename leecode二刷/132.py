class Solution:
    def minCut(self, s: str) -> int:
        n = len(s)
        dp = [[True] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):  # n-1 0
            for j in range(i + 1, n):
                dp[i][j] = ((s[i] == s[j]) and dp[i + 1][j - 1])
        ans = [float("inf")] * n
        for r in range(n):
            if dp[0][r]:#如果字符0-r 是回文 则为 分割次数 = 0
                ans[r] = 0
            else:
                for l in range(r):
                    if dp[l + 1][r]:
                        ans[r] = min(ans[r], ans[l] + 1)

        return ans[-1]


if __name__ == '__main__':
    s = "aaabaa"
    # a aabaa
    # aabaa a
    solution = Solution()
    print(solution.minCut(s))
