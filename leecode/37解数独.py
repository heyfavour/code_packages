from typing import List


class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        self.sudu_len = len(board)

        def back(i, j, board):
            if not self.checksudu(i, j, board): return False
            newi, newj = self.get_next_ij(i, j, board)
            if newi > self.sudu_len - 1: return True
            for num in self.get_able_nums(newi, newj, board):
                board[newi][newj] = num
                if back(newi, newj, board): return board
                board[newi][newj] = "."
            return False

        return back(0, 0, board)

    def checksudu(self, i, j, board):
        # board[i]
        # 横向
        if len([x for x in board[i] if x != '.']) != len(set([x for x in board[i] if x != '.'])):
            return False
        if len([board[i][j] for i in range(self.sudu_len) if board[i][j] != '.']) != len(
                set([board[i][j] for i in range(self.sudu_len) if board[i][j] != '.'])):
            return False
        k = [board[x + (j // 3) * 3][y + (i // 3) * 3] for x in range(3) for y in range(3) if
             board[x + (j // 3) * 3][y + (i // 3) * 3] != "."]
        if len(k) != len(set(k)): return False
        return True

    def get_next_ij(self, i, j, board):
        if board[i][j] == '.':return i,j
        if j == self.sudu_len - 1:
            i = i + 1
            j = 0
        else:
            j = j + 1
        if i > self.sudu_len - 1: return (i, j)
        if board[i][j] != '.':
            return self.get_next_ij(i, j, board)
        return i, j

    def get_able_nums(self, i, j, board):
        exist_nums = [x for x in board[i] if x != "."] + [board[x][j] for x in range(self.sudu_len) if
                                                          board[x][j] != '.'] + [
                         board[y + (i // 3) * 3][x + (j // 3) * 3] for x in range(3) for y in range(3) if
                         board[y + (i // 3) * 3][x + (j // 3) * 3] != "."]
        exist_nums = set(exist_nums)
        able_nums = [str(i) for i in range(1, 10) if str(i) not in exist_nums]
        return able_nums


if __name__ == '__main__':
    s = Solution()
    board = [["5", "3", ".", ".", "7", ".", ".", ".", "."],
             ["6", ".", ".", "1", "9", "5", ".", ".", "."],
             [".", "9", "8", ".", ".", ".", ".", "6", "."],
             ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
             ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
             ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
             [".", "6", ".", ".", ".", ".", "2", "8", "."],
             [".", ".", ".", "4", "1", "9", ".", ".", "5"],
             [".", ".", ".", ".", "8", ".", ".", "7", "9"]
             ]
    board = [[".", ".", "9", "7", "4", "8", ".", ".", "."], ["7", ".", ".", ".", ".", ".", ".", ".", "."],
             [".", "2", ".", "1", ".", "9", ".", ".", "."], [".", ".", "7", ".", ".", ".", "2", "4", "."],
             [".", "6", "4", ".", "1", ".", "5", "9", "."], [".", "9", "8", ".", ".", ".", "3", ".", "."],
             [".", ".", ".", "8", ".", "3", ".", "2", "."], [".", ".", ".", ".", ".", ".", ".", ".", "6"],
             [".", ".", ".", "2", "7", "5", "9", ".", "."]]
    print(s.solveSudoku(board))
