class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        while "**" in p:
            p = p.replace("**", "*")
        m, n = len(s), len(p)
        dp = [[False] * (n+1) for _ in range(m + 1)]
        dp[0][0] = True
        if p[0] == "*":dp[0][1] = True
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                print(i,j)
                if p[j - 1] == "?" or (s[i - 1] == p[j - 1]):
                    dp[i][j] = dp[i - 1][j - 1]
                if p[j - 1] == "*": dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
        return dp[-1][-1]

if __name__ == '__main__':
    solution = Solution()
    s = "babbbbaabababaabbababaababaabbaabababbaaababbababaaaaaabbabaaaabababbabbababbbaaaababbbabbbbbbbbbbaabbb"
    p = "b**bb**a**bba*b**a*bbb**aba***babbb*aa****aabb*bbb***a"
    print(solution.isMatch(s, p))
