class Solution:
    def combine(self, n: int, k: int) -> list[list[int]]:
        ans = []

        def dfs(L, i):
            if len(L) == k:
                ans.append(L.copy())
                return
            for j in range(i, n + 1):
                L.append(j)
                dfs(L, j + 1)
                L.pop()

        dfs([], 1)
        return ans


if __name__ == '__main__':
    n = 4
    k = 2
    solution = Solution()
    print(solution.combine(n,k))
