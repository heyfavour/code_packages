class Solution:
    def spiralOrder(self, matrix: list[list[int]]) -> list[int]:
        h = len(matrix)
        w = len(matrix[0])

        def backtrack(i):
            w_left = w - 2 * i
            h_left = h - 2 * i
            if w_left <= 0 or h_left <= 0: return []
            if h_left == 1: return matrix[i][i:i + w_left]
            if w_left == 1: return [matrix[row][i] for row in range(i, i + h_left)]
            res = []
            res = res + matrix[i][i:i + w_left]
            res = res + [matrix[row][i + w_left - 1] for row in range(i + 1, i + h_left - 1)]
            res = res + matrix[i + h_left - 1][i:i + w_left][::-1]
            res = res + [matrix[row][i] for row in range(i + 1, i + h_left - 1)][::-1]


            return res + backtrack(i + 1)

        return backtrack(0)


if __name__ == '__main__':
    solution = Solution()
    matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    print(solution.spiralOrder(matrix))
