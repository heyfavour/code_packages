class Solution:
    def solve(self, board: list[list[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        m, n = len(board), len(board[0])
        board = [["P" if j == "O" else j for j in board[i]] for i in range(m)]
        dir = [(0, 1), (0, -1), (-1, 0), (1, 0)]

        def check_dir(i, j):
            if not (0 <= i < m and 0 <= j < n) or board[i][j] in ("X", "O"): return
            board[i][j] = "O"
            for dx, dy in dir:
                ni, nj = i + dx, j + dy
                check_dir(ni, nj)

        for i in range(m):
            for j in range(n):
                if (i == 0 or i == m - 1 or j == 0 or j == n - 1) and board[i][j] == "P":
                    check_dir(i, j)
        return [["X" if j == "P" else j for j in board[i]] for i in range(m)]


if __name__ == '__main__':
    board = [["X", "X", "X", "X"], ["X", "O", "O", "X"], ["X", "X", "O", "X"], ["X", "O", "X", "X"]]

    solution = Solution()
    print(solution.solve(board))
