class Solution:
    """
    DFS
    def knightProbability(self, n: int, k: int, row: int, column: int) -> float:
        position = (column, row)
        walk_out = 0
        steps = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
        def _walk(position, walk_k):
            nonlocal walk_out
            if position[0] < 0 or position[0] >= n or position[1] < 0 or position[1] >= n: return
            print(position,walk_k)
            if walk_k == k:
                walk_out = walk_out + 1
                return True
            for step in steps:
                x, y = position[0], position[1]
                new_position = (x + step[0], y + step[1])
                _walk(new_position, walk_k + 1)
        _walk(position, 0)
        return walk_out * 1.0 / (8 ** k)
    """

    def knightProbability(self, n: int, k: int, row: int, column: int) -> float:
        dirs = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
        dp = [[[0] * n for _ in range(n)] for _ in range(k + 1)]  # dp[k+1][i][j]
        for step in range(k + 1):
            for i in range(n):
                for j in range(n):
                    if step == 0:
                        dp[step][i][j] = 1
                    else:
                        for x, y in dirs:
                            next_i = i + x
                            next_j = j + y
                            if 0 <= next_i < n and 0 <= next_j < n:# 8个方向 的概率 相加
                                #下一步的概率 = 下一步的概率 上一步的或者且能走到你
                                dp[step][i][j] = dp[step][i][j]  + dp[step - 1][next_i][next_j] / 8
                    print(dp)

        return dp[k][row][column]


if __name__ == '__main__':
    solution = Solution()
    p = solution.knightProbability(3, 2, 0, 0)
    print(p)
