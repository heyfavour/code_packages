class Solution:
    def setZeroes(self, matrix: list[list[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m, n = len(matrix), len(matrix[0])
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    for ti in range(m):
                        if matrix[ti][j] != 0: matrix[ti][j] = "."
                    for tj in range(n):
                        if matrix[i][tj] != 0: matrix[i][tj] = "."
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == ".": matrix[i][j] = 0
        return matrix


if __name__ == '__main__':
    solution = Solution()
    matrix = [[0, 1, 2, 0], [3, 4, 5, 2], [1, 1, 1, 5]]
    print(solution.setZeroes(matrix))
