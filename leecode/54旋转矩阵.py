from typing import List


class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:

        def helper(depth):
            row_left = len(matrix) - depth * 2
            col_left = len(matrix[0]) - depth * 2
            if row_left <= 0 or col_left <= 0: return []
            if row_left == 1: return matrix[depth][depth:depth + col_left]
            if col_left == 1: return [matrix[i][depth] for i in range(depth, depth + row_left)]

            res = []
            res = res + matrix[depth][depth:depth + col_left]
            res = res + [matrix[i][depth + col_left - 1] for i in range(depth + 1, depth + row_left - 1)]
            res = res + matrix[depth + row_left - 1][depth:depth + col_left][::-1]
            res = res + [matrix[i][depth] for i in range(depth + row_left - 2, depth, -1)]

            return res + helper(depth + 1)

        res = helper(0)
        return res


if __name__ == '__main__':
    L = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]
    s = Solution()
    print(s.spiralOrder(L))
