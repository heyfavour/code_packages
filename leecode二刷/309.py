class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        buy = []
        """
        f[i][0] = max(f[i-1][2],f[i-1][0])#啥都没有 冷冻期 或者啥都没有
        f[i][1] = max(f[i-1][1],f[i-1][0]-prices[i])#有一只股票
        f[i][2] = f[i-1][1]+prices[0]
        """
        if not prices:return 0
        dp = [[0,0,0] for _ in range(len(prices))]
        dp[0][1] = -prices[0]
        for i in range(1,len(prices)):
            dp[i][0] = max(dp[i - 1][2], dp[i - 1][0])  # 啥都没有 冷冻期 或者啥都没有
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])  # 有一只股票
            dp[i][2] = dp[i - 1][1] + prices[i]
        print(dp)
        return max(dp[-1])



if __name__ == '__main__':
    solution = Solution()
    prices = [1, 2, 3, 0, 2]
    print(solution.maxProfit(prices))
