class Solution:
    def solveSudoku(self, board: list[list[str]]) -> None:
        def check(i, j, board):
            row_nums = [board[i][col] for col in range(9) if board[i][col] != "."]
            if len(row_nums) != len(set(row_nums)): return False
            col_nums = [board[row][j] for row in range(9) if board[row][j] != '.']
            if len(col_nums) != len(set(col_nums)): return False
            zone_nums = [board[(i // 3) * 3 + row][(j // 3) * 3 + col] for row in range(3) for col in range(3) if
                         board[(i // 3) * 3 + row][(j // 3) * 3 + col] != '.']
            if len(zone_nums) != len(set(zone_nums)): return False
            return True

        def get_coordinate(i, j, board):
            if board[i][j] == ".": return i, j
            if j <= 7:
                j = j + 1
            else:
                i = i + 1
                j = 0
            if i == 9: return i, j
            if board[i][j] != ".": return get_coordinate(i, j, board)
            return i, j

        def get_able_nums(i, j, board):
            row_nums = [board[i][col] for col in range(9) if board[i][col] != "."]
            col_nums = [board[row][j] for row in range(9) if board[row][j] != '.']
            zone_nums = [board[(i // 3) * 3 + row][(j // 3) * 3 + col] for row in range(3) for col in range(3) if
                         board[(i // 3) * 3 + row][(j // 3) * 3 + col] != '.']
            nums = set(row_nums + col_nums + zone_nums)
            return [str(i) for i in range(1, 10) if str(i) not in nums]

        def back(i, j, board):
            # 剪枝 是否为可用的路径
            if not check(i, j, board): return False
            i, j = get_coordinate(i, j, board)  # 确认坐标
            if i == 9: return True
            for num in get_able_nums(i, j, board):  # 可用的数
                board[i][j] = num
                if back(i, j, board): return board
                board[i][j] = "."
            return False

        return back(0, 0, board)


if __name__ == '__main__':
    solution = Solution()
    board = [["5", "3", ".", ".", "7", ".", ".", ".", "."],
             ["6", ".", ".", "1", "9", "5", ".", ".", "."],
             [".", "9", "8", ".", ".", ".", ".", "6", "."],
             ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
             ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
             ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
             [".", "6", ".", ".", ".", ".", "2", "8", "."],
             [".", ".", ".", "4", "1", "9", ".", ".", "5"],
             [".", ".", ".", ".", "8", ".", ".", "7", "9"]]

    print(solution.solveSudoku(board))
