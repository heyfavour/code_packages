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

    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[True]*n for i in range(n)]
        max_len,begin = 1,0
        for i in range(n-1,-1,-1):#起始位置
            for j in range(i+1,n):#终点
                dp[i][j] = (s[i]==s[j]) and dp[i+1][j-1]
                L = j-i+1
                if dp[i][j] and L>max_len:
                    max_len=L
                    begin = i
        return s[begin:begin+max_len]


if __name__ == '__main__':
    solution = Solution()
    s = "babad"
    max_s = solution.longestPalindrome(s)
    print(max_s)
