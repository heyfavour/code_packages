"""
01背包
max_weight 4
物品  重量  价值
0    1     14
1    3     20
2    4     30

dp[i][j] dp[物品][容量] 考虑物品i下容量的最大价值
dp[i][j] = max(dp[i-1][j] #不放物品i dp[i-1][j-weight[i]]+value[i])#放物品i

"""

class Solution:
    def numSquares(self, n: int) -> int:
        pass

if __name__ == '__main__':
    solution = Solution()
    print(solution(13))