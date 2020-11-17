from typing import List


class Solution:
    @classmethod
    def maxProfit(self, prices: List[int]) -> int:
        buy = -1
        result = 0
        for i in range(len(prices)):
            if (i == len(prices)-1):#卖出条件
                if buy>=0:result = result + (prices[-1] - prices[buy])
            elif prices[i] > prices[i + 1] and buy >= 0:#卖出条件
                result = result + (prices[i]-prices[buy])
                buy = -1
            elif prices[i] > prices[i + 1] and buy < 0:#不操作1
                continue
            elif prices[i] < prices[i + 1] and buy >= 0:#不操作2
                continue
            elif  prices[i]<prices[i + 1] and buy<0:#买入条件
                buy = i
        return result



if __name__ == '__main__':
    L = [7, 1, 5, 3, 6, 4]
    L = [2,4,1]
    print(Solution.maxProfit(L))
