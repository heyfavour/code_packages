class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0]*n
        for i in range(n):
            dp[1] = 1
            dp[2] = 2
            dp[3] = dp[2] +

if __name__ == '__main__':
    pass
