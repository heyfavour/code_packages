#完全背包问题
class Solution:
    def numSquares(self, n: int) -> int:
        weight,i = [],0
        while i ** 2 <= n:
            weight.append(i ** 2)
            i = i + 1
        dp = [[0] *(n+1)  for _ in range(len(weight))]  # [i][j] i物品 j容量 j
        #容量为0 都是0 dp[i][0] = 0
        for i in range(1,len(weight)):#物品 nums
            for j in range(1,n+1):#容量
                if j<weight[i]:#容量不够
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = min(dp[i-1][j],dp[i][j-weight[i]]+1)
        return dp[-1][-1]
if __name__ == '__main__':
    solution = Solution()
    n = 13
    print(solution.numSquares(n))
