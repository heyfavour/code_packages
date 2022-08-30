# 背包问题
'''
输入：背包容量C，物品个数num，以及每个物品的重量w[i],价值v[i]
输出：当背包容量为C时候的最大价值
置dp[0][j]为0，j<=C
对于第i个物品，i<=num,做如下循环
    对于容量j从0开始，j<=C,做如下循环
        如果j<w[i],则dp[i][j]=dp[i-1][j]
        否则，dp[i][j]=max{dp[i-1][j],dp[i-1][j-w[i]]+v[i]}
返回dp[num][C]即为当背包容量为C时候的最大值
'''
# dp[i][j] = max(  dp[i-1][j],dp[i-1][j-weight[i]+value[i] )
# dp[i][j] = max(  dp[i-1][j],dp[i][j-weight[i]+value[i] )

