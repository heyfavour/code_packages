from typing import List


class Solution:
    @classmethod
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix[0])
        for i in range(n):
            for j in range(n):
                if i == j: break
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for i in range(n):
            for j in range(n):
                if j >= n / 2: break
                matrix[i][j], matrix[i][n - 1 - j] = matrix[i][n - 1 - j], matrix[i][j]
        return matrix


if __name__ == '__main__':
    L = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    print(Solution.rotate(L))
