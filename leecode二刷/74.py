class Solution:
    def searchMatrix(self, matrix: list[list[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        i, j = 0, n - 1
        while 0 <= i <= m - 1 and 0 <= j <= n - 1:
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] < target:
                i = i + 1
            elif matrix[i][j] > target:
                j = j - 1
        return False


if __name__ == '__main__':
    matrix = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]]
    target = 6
    solution = Solution()
    print(solution.searchMatrix(matrix, target))
