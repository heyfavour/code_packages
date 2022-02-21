class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        if n <= 1: return s
        dp = [[False] * n for _ in range(n)]
        for i in range(n): dp[i][i] = True
        max_len = 1
        begin = 0
        for L in range(2, n + 1):
            for i in range(n):
                j = i + L - 1
                if j >= n: break
                if s[i] != s[j]:
                    dp[i][j] = False
                else:
                    if L <= 3:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]
                if dp[i][j] and max_len < L:
                    max_len = L
                    begin = i
        return s[begin:begin + max_len]


if __name__ == '__main__':
    solution = Solution()
    s = "babad"
    max_s = solution.longestPalindrome(s)
    print(max_s)
