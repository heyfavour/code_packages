class Solution:
    def generate(self, numRows: int) -> list[list[int]]:
        #1
        #1 1
        #1 2 1
        #1 3 3 1
        dp = [[1]*(i+1) for i in range(numRows)]
        for i in range(numRows):
            for j in range(i):
                if j == 0:continue
                dp[i][j] = dp[i-1][j-1]+dp[i-1][j]

        return dp