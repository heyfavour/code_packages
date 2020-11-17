from typing import List


class Solution:
    @classmethod
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(cost)>sum(gas):return -1
        n = len(gas)
        road = [0]*n
        for i in range(len(gas)):
            ava = 0
            for j in range(len(gas)):
                ava = ava + gas[(i+j)%n] - cost[(i+j)%n]
                if ava<0:break
            if ava>=0:return i
        return -1



if __name__ == '__main__':
    gas = [1, 2, 3, 4, 5]
    cost = [3, 4, 5, 1, 2]
    print(Solution.canCompleteCircuit(gas, cost))
