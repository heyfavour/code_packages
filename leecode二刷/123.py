class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        n = len(prices)
        buy1, profit1, buy2, profit2 = prices[0], 0, prices[0], 0
        for i in range(n):
            buy1 = min(buy1, prices[i])
            profit1 = max((profit1, prices[i] - buy1))
            buy2 = min(buy2, prices[i]-profit1)
            profit2 = max(profit2, prices[i]-buy2)
            print(buy1, profit1, buy2, profit2)
        return profit2


if __name__ == '__main__':
    solutin = Solution()
    prices = [3, 3, 5, 0, 0, 3, 1, 4]
    print(solutin.maxProfit(prices))
