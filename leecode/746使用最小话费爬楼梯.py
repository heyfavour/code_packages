from typing import List


class Solution:
    @classmethod
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        dp = [0] * len(cost)
        for i in range(len(cost)):
            if i in (0, 1):
                dp[i] = 0
            else:
                dp[i] =  min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        return dp[-1]

if __name__ == '__main__':
    cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
    print(Solution.minCostClimbingStairs(cost))
