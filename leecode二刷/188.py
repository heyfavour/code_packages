class Solution:
    def maxProfit(self, k: int, prices: list[int]) -> int:
        n = k // 2
        buy = [float("inf")]*n
        profit = [float("-inf")]*n
        for i in range(len(k)):
            buy[0] = min(buy[0],prices[i])#支出
            profit[0] = max(profit[0],profit[0]-buy[0])#收入
            for i in range(1,n+1):
                buy[i] = min(buy[i],profit[i]-prices[i])


if __name__ == '__main__':
    solution = Solution()
    k = 2
    prices = [2, 4, 1]
    print(solution.maxProfit(k, prices))
