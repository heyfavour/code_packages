class Solution:
    def canCompleteCircuit(self, gas: list[int], cost: list[int]) -> int:
        n = len(gas)
        if sum(gas) < sum(cost): return -1
        oil = 0
        start = 0
        for i in range(n):
            oil = gas[i] - cost[i] + oil
            if oil < 0:
                oil = 0
                start = i + 1
        return start


if __name__ == '__main__':
    solution = Solution()
    gas = [1, 2, 3, 4, 5]
    cost = [3, 4, 5, 1, 2]
    print(solution.canCompleteCircuit(gas, cost))
