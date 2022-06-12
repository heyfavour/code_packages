class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        m,n = len(s),len(t)
        dp = [[0]*(m+1) for i in range(n+1)]#dp[t][s]
        for j in range(m+1):dp[0][j] = 1
        for i in range(1,n+1):#字符 t
            for j in range(1,m+1):#字符 s
                if t[i-1]==s[j-1]:
                    dp[i][j] = dp[i-1][j-1]+dp[i][j-1]#前一位+ s的i位在本轮中产生的个数
                else:
                    dp[i][j] = dp[i][j-1]
        return dp[-1][-1]



if __name__ == '__main__':
    solution = Solution()
    s = "rabbbit"
    t = "rabbit"
    print(solution.numDistinct(s,t))