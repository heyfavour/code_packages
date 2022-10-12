# 01背包
def pageckage_01():
    w = [3, 4, 5, 7]
    v = [1, 5, 6, 9]
    pw = 10
    w.insert(0, 0)
    v.insert(0, 0)

    dp = [[0] * (pw + 1) for _ in range(len(w))]

    for i in range(1, len(w)):  # 物品
        for j in range(1, pw + 1):
            if j < w[i]:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w[i]] + v[i])
    print(dp)

    def get_path(dp, w, pw):
        i = len(w) - 1
        j = pw
        res = []
        while i != 0 and j != 0:
            if dp[i][j] == dp[i - 1][j]:
                i = i - 1
            else:
                res.append(i - 1)
                j = j - w[i]
                i = i - 1
        return res

    get_path(dp, w, pw)


def pageckage_complete():
    w = [3, 4, 5, 7]
    v = [1, 5, 6, 9]
    pw = 10
    w.insert(0, 0)
    v.insert(0, 0)

    dp = [[0] * (pw + 1) for _ in range(len(w))]

    for i in range(1, len(w)):  # 物品
        for j in range(1, pw + 1):
            if j < w[i]:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - w[i]] + v[i])

    print(dp)

    def get_path(dp, w, c):
        i = len(w) - 1
        j = c
        res = []
        while i != 0 and j != 0:
            if dp[i][j] == dp[i - 1][j]:
                i = i - 1
            else:
                res.append(i - 1)
                j = j - w[i]
        return res

    get_path(dp, w, pw)


def package_multi():
    w = [3, 4, 5, 7]
    v = [1, 5, 6, 9]
    n = [1, 2, 3, 4]
    pw = 10
    w.insert(0, 0)
    v.insert(0, 0)
    n.insert(0, 0)
    dp = [[0] * (pw + 1) for i in range(len(w))]
    for i in range(1, len(w)):
        count = [0 for i in range(pw + 1)]  # 记录i已经放了多少个
        for j in range(1, pw + 1):
            if j < w[i]:
                dp[i][j] = dp[i - 1][j]
            else:
                if count[j - w[i]] < n[i]:  # 还能放i
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - w[i]] + v[i])
                    count[j] = count[j - w[i]] + 0 if dp[i][j] == dp[i - 1][j - 1] else 1
                else:  # 不能放i
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - n[i] * w[i]] + n[i] * v[i])
                    count[j] = n[i]
    print(dp)


def package_01_one_dim():
    c = 10
    w = [3, 4, 5, 7]
    v = [1, 5, 6, 9]
    w.insert(0, 0)
    v.insert(0, 0)
    dp = [0] * (c + 1)  # dp[背包]
    for i in range(1, len(w)):
        for j in range(c, 0, -1):
            if w[i] <= j: dp[j] = max(dp[j], dp[j - w[i]] + v[i])
    print(dp)
    return dp[-1]


def pageckage_complete_one_dim():
    c = 10
    w = [3, 4, 5, 7]
    v = [1, 5, 6, 9]
    w.insert(0, 0)
    v.insert(0, 0)
    dp = [0] * (c + 1)  # dp[背包]
    for i in range(1, len(w)):
        for j in range(0, c + 1):
            if w[i] <= j: dp[j] = max(dp[j], dp[j - w[i]] + v[i])
    print(dp)
    return dp[-1]


if __name__ == '__main__':
    pageckage_01()
    pageckage_complete()
    package_multi()
    print("===========优化===============")
    package_01_one_dim()
    pageckage_complete_one_dim()
