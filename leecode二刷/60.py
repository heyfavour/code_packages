class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        ans = 0

        def helper(L, TL):
            nonlocal ans
            if len(TL) == n:
                ans = ans + 1
                if ans == k: return TL
            for i, v in enumerate(L):
                L.pop(i)
                TL.append(v)
                ans_list = helper(L, TL)
                if ans_list: return ans_list
                TL.pop()
                L.insert(i, v)

        return helper(list(range(1, n + 1)), [])


if __name__ == '__main__':
    solution = Solution()
    n = 4
    k = 9
    print(solution.getPermutation(n, k))
