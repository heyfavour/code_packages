class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        n = len(prices)
        ans = 0
        for i in range(1,n):
            if prices[i] > prices[i-1]:
                ans = ans + prices[i] - prices[i-1]
        return ans
if __name__ == '__main__':
    solution = Solution()
    prices = [7, 1, 5, 3, 6, 4]
    print(solution.maxProfit(prices))