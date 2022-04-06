class Solution:
    def exist(self, board: list[list[str]], word: str) -> bool:
        dirs = ((-1, 0), (0, 1), (1, 0), (0, -1))
        _len = len(word)
        m, n = len(board), len(board[0])

        def dfs(i, j, c):
            if board[i][j] != word[c]: return False
            if c == _len - 1: return True
            for x, y in dirs:
                nx, ny = x + i, y + j
                tc = board[i][j]
                if not (0 <= nx < m and 0 <= ny < n): continue
                board[i][j] = "."
                ans = dfs(nx, ny, c + 1)
                board[i][j] = tc
                if ans: return True

        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0): return True
        return False


if __name__ == '__main__':
    solution = Solution()
    board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
    word = "ABCCED"
    board = [["C", "A", "A"], ["A", "A", "A"], ["B", "C", "D"]]
    word = "AAB"
    print(solution.exist(board, word))
