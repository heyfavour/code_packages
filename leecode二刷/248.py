class Solution:
    def strobogrammaticInRange(self, low: str, high: str) -> int:
        m, n = len(low), len(high)
        if int(low) > int(high): return 0
        dp = [[n] for _ in range(max(n, 3))]
        dp[0] = ["1", "8", "0"]
        dp[1] = ["11", "88", "96", "69"]

        def gen(L, add):
            mid = len(L[0]) >> 1
            ans = []
            for i in L:
                for m in add:
                    ans.append(i[:mid] + m + i[mid:])
            return ans

        for i in range(n):
            if i in (0, 1): continue
            if i % 2 == 1:  # 偶数
                dp[i] = gen(dp[i - 2], ["00", "11", "88", "96", "69"])
            else:
                dp[i] = gen(dp[i - 1], ["1", "0", "8"])
        count = 0
        if m == n:
            for i in dp[m - 1]:
                if int(low) <= int(i) <= int(high):
                    count = count + 1
            return count
        for i in range(m - 1, n):
            if i == m - 1:
                for num in dp[i]:
                    if int(num) >= int(low): count = count + 1
            elif i == n - 1:
                for num in dp[i]:
                    if int(num) <= int(high): count = count + 1
            else:
                count = count + len(dp[i])
        return count


