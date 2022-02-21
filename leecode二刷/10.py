class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m,n= len(s),len(p)
        def match(i,j):
            pass
        dp = [[False]*(n+1) for _ in range(m+1)]
        dp[0][0] = True
        for i in range(m+1):
            for j in range(1,n+1):
                if p[j-1] == "*":
                    dp[i][j] == 
