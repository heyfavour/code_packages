from typing import List


class Solution:
    @classmethod
    def maxProfit(self, prices: List[int]) -> int:
        result = []
        buy = -1
        for i in range(len(prices)):
            if i == len(prices) - 1:  # 卖出
                if buy >= 0:
                    result.append(prices[i] - prices[buy])
            elif prices[i] < prices[i + 1] and buy < 0:
                buy = i
            elif prices[i] > prices[i + 1] and buy >= 0:
                result.append(prices[i] - prices[buy])
                buy = -1
        if len(result)>=2:
            result.sort()
            return sum(result[-1:-3:-1])
        return sum(result)


if __name__ == '__main__':
    L = [3, 3, 5, 0, 0, 3, 1, 4]
    L = [1,4,2]
    print(Solution.maxProfit(L))
