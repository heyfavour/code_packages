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

优化至1维
置dp[0][j]为0，j<=C
对于第i个物品，i<=num,做如下循环
    对于容量j从0开始，j<=C,做如下循环

'''


# dp[i][j] = max(  dp[i-1][j],dp[i-1][j-weight[i]+value[i] )
# dp[i][j] = max(  dp[i-1][j],dp[i][j-weight[i]+value[i] )

def package():
    c = 10
    w = [3, 4, 5, 7]
    v = [1, 5, 6, 9]
    w.insert(0,0)
    v.insert(0,0)
    dp = [[0] * (c + 1) for _ in range(len(w))]#dp[物品][背包]
    for i in range(1,len(w)):
        for j in range(1,c+1):
            if w[i]>j:dp[i][j] = dp[i-1][j]
            else:dp[i][j] = max(dp[i-1][j],dp[i-1][j-w[i]]+v[i])
    import pprint
    pprint.pprint(dp)
    return dp[-1][-1]

def package_one_dim():
    c = 10
    w = [3, 4, 5, 7]
    v = [1, 5, 6, 9]
    w.insert(0,0)
    v.insert(0,0)
    dp = [0] * (c + 1)#dp[背包]
    for i in range(1,len(w)):
        for j in range(c,0,-1):
            if w[i]<=j:dp[j] = max(dp[j],dp[j-w[i]]+v[i])
    print(dp)
    return dp[-1]


def package2():
    c = 10
    w = [3, 4, 5, 7]
    v = [1, 5, 6, 9]
    w.insert(0,0)
    v.insert(0,0)
    dp = [[0] * (c + 1) for _ in range(len(w))]#dp[物品][背包]
    for i in range(1,len(w)):
        for j in range(1,c+1):
            if w[i]>j:dp[i][j] = dp[i-1][j]
            else:dp[i][j] = max(dp[i-1][j],dp[i][j-w[i]]+v[i])
    import pprint
    pprint.pprint(dp)
    return dp[-1][-1]

def package2_one_dim():
    c = 10
    w = [3, 4, 5, 7]
    v = [1, 5, 6, 9]
    w.insert(0,0)
    v.insert(0,0)
    dp = [0] * (c + 1)#dp[背包]
    for i in range(1,len(w)):
        for j in range(c,0,-1):
            if w[i]<=j:dp[j] = max(dp[j],dp[j-w[i]]+v[i])
    print(dp)
    return dp[-1]


if __name__ == '__main__':
    print(package())
    print(package_one_dim())
    print(package2())
    print(package2_one_dim())
