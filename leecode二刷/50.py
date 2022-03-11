class Solution:
    def myPow(self, x: float, n: int) -> float:
        def dfs(n):
            if n == 0: return 1
            y = dfs(n // 2)
            return y * y if n % 2 == 0 else y * y * x

        return dfs(n) if n >= 0 else 1 / dfs(-n)


if __name__ == '__main__':
    solution = Solution()
    x = 2
    n = 10
    print(solution.myPow(x, n))
