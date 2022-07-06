class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        def dfs(x):
            if x == 1: return True
            if x <= 1: return False
            return int(x / 2) == x / 2 and dfs(x / 2)

        return dfs(n)