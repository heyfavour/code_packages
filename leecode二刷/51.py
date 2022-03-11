import copy


class Solution:
    def solveNQueens(self, n: int) -> list[list[str]]:
        ans = []
        matrix = [[0] * n for i in range(n)]
        _dict = {0:".",1:"Q"}

        def check(matrix, i, j):
            directions = ((-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0))
            all = [matrix[i + x * move][j + y * move] for move in range(n) for x, y in directions
                   if 0 <= i + x * move <= n - 1 and 0 <= j + y * move <= n - 1]
            if sum(all) == 0: return True
            return False

        def backtrtack(matrix, row, queen_num):
            if queen_num == n:
                ans.append([_dict[matrix[i][j]] for i in range(n) for j in range(n)])
                return
            for j in range(n):
                if not check(matrix, row, j): continue
                matrix[row][j] = 1
                backtrtack(matrix, row + 1, queen_num + 1)
                matrix[row][j] = 0
            return False

        backtrtack(matrix, 0, 0)

        return ans


if __name__ == '__main__':
    solution = Solution()
    n = 4
    print(solution.solveNQueens(n))
