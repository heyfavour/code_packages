from typing import List


class Solution:
    # 前i天的最大收益 = max{前i-1天的最大收益，第i天的价格-前i-1天中的最小价格}
    @classmethod
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [0] * n
        for i in range(n):
            if i == 0:
                dp[i] = 0
            else:
                dp[i] = max([
                    dp[i - 1], prices[i] - min(prices[:i])
                ])
        return max(dp)


if __name__ == '__main__':
    L = [7, 1, 5, 3, 6, 4]
    L = [7,6,4,3,1]
    print(Solution.maxProfit(L))
