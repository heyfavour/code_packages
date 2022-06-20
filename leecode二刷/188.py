class Solution:
    def maxProfit(self, k: int, prices: list[int]) -> int:
        n = len(prices)
        k = min(k,n//2)
        if k ==0:return 0

        profit_buy = [float("-inf")] * k
        profit_sell = [0] * k
        for i in range(n):
            for j in range(k):
                profit_buy[j] = max(profit_buy[j], (0 if j ==0 else profit_sell[j-1]) - prices[i])
                profit_sell[j] = max(profit_sell[j], profit_buy[j] + prices[i])

        return profit_sell[-1]


if __name__ == '__main__':
    solution = Solution()
    k = 2
    prices = [3,2,6,5,0,3]
    print(solution.maxProfit(k, prices))
