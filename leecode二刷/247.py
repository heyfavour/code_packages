class Solution:
    def findStrobogrammatic(self, n: int) -> List[str]:
        nums_dict = {"0": "0", "1": "1", "8": "8", "9": "6", "6": "9"}
        candidate_ji = ["11", "88", "69", "96", "00"]
        candidate_ou = ["0", "1", "8"]
        dp = [[] for _ in range(n)]
        dp[0] = ["0", "1", "8"]
        if n == 1: return dp[0]
        dp[1] = ["11", "88", "69", "96"]
        if n == 2: return dp[1]

        def gen(L, candidate):
            ans = []
            mid = len(L[0]) >> 1
            for num in L:
                for i in candidate:
                    ans.append(num[:mid] + i + num[mid:])
            return ans

        for i in range(n):
            if i in (0, 1): continue
            if (i + 1) % 2 == 0:
                candidate_num = i - 2
                candidate = candidate_ji
            else:
                candidate_num = i - 1
                candidate = candidate_ou
            dp[i] = gen(dp[candidate_num], candidate)
        return dp[-1]
