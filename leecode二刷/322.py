class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        coins.insert(0,0)
        dp = [float("inf")]*(amount+1)
        dp[0] = 0
        for coin in coins:
            for j in range(1,amount+1):
                if j>=coin:dp[j] = min(dp[j],dp[j-coin]+1)
        return -1 if dp[-1] == float("inf") else dp[-1]

if __name__ == '__main__':
    coins = [1, 2, 5]
    amount = 11
    solution = Solution()
    print(solution.coinChange(coins,amount))