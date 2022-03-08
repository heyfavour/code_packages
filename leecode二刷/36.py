class Solution:
    def isValidSudoku(self, board: list[list[str]]) -> bool:
        n = len(board)
        row = {i: {} for i in range(n)}
        col = {i: {} for i in range(n)}
        zone = {i: {j: {} for j in range(n // 3)} for i in range(n // 3)}
        for i in range(n):
            for j in range(n):
                val = board[i][j]
                if val== ".":continue
                row[i][val] = row[i].get(val,0) + 1
                col[j][val] = col[j].get(val,0) + 1
                zone[i//3][j//3][val] =  zone[i//3][j//3].get(val,0) + 1
                if row[i][val]>1 or col[j][val]>1 or zone[i//3][j//3][val]>1:return False
        return True


if __name__ == '__main__':
    board = [
        ["5", "3", ".", ".", "7", ".", ".", ".", "."]
        , ["6", ".", ".", "1", "9", "5", ".", ".", "."]
        , [".", "9", "8", ".", ".", ".", ".", "6", "."]
        , ["8", ".", ".", ".", "6", ".", ".", ".", "3"]
        , ["4", ".", ".", "8", ".", "3", ".", ".", "1"]
        , ["7", ".", ".", ".", "2", ".", ".", ".", "6"]
        , [".", "6", ".", ".", ".", ".", "2", "8", "."]
        , [".", ".", ".", "4", "1", "9", ".", ".", "5"]
        , [".", ".", ".", ".", "8", ".", ".", "7", "9"]]

    solution = Solution()
    print(solution.isValidSudoku(board))
