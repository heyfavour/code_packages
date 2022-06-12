class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        n = len(prices)
        min_price, max_profit = prices[0], 0
        for i in range(n):
            max_profit = max(prices[i] - min_price, max_profit)
            min_price = min(prices[i], min_price)
        return max_profit


if __name__ == '__main__':
    prices = [7, 1, 5, 3, 6, 4]
    solution = Solution()
    print(solution.maxProfit(prices))
