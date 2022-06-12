class Solution:
    def partition(self, s: str) -> list[list[str]]:
        n = len(s)
        dp = [[True] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):  # n-1 ->0
            for j in range(i + 1, n):
                dp[i][j] = (s[i] == s[j]) and dp[i + 1][j - 1]

        ans, one_path = [], []

        def dfs(start):
            if start == n:
                ans.append(one_path.copy())
                return
            for end in range(start, n):
                if dp[start][end]:
                    one_path.append(s[start:end + 1])
                    dfs(end + 1)
                    one_path.pop()

        dfs(0)
        return ans


if __name__ == '__main__':
    solution = Solution()
    s = "aab"
    print(solution.partition(s))
