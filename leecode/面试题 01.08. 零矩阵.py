from typing import List
import numpy as np


class Solution:
    @classmethod
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix)
        n = len(matrix[0])

        def helper(i, j):
            matrix[i] = [0 if num == 0 else -1 for num in matrix[i]]
            for row in range(m):
                matrix[row][j] = 0 if matrix[row][j] == 0 else -1

        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0: helper(i, j)
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == -1: matrix[i][j] = 0
        return matrix


if __name__ == '__main__':
    L = [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]
    print(Solution.setZeroes(L))
