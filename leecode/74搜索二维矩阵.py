from typing import List


class Solution:
    @classmethod
    def searchMatrix2(self, matrix: List[List[int]], target: int) -> bool:
        # 两次查找 超时
        # 二维变一维
        l, r = 0, len(matrix) * len(matrix[0]) - 1
        while l <= r:
            mid = (l + r) // 2  # mid = 5
            w_1 = mid // len(matrix[0])
            w_2 = mid % len(matrix[0])
            if matrix[w_1][w_2] == target:
                return True
            elif matrix[w_1][w_2] < target:
                l = mid + 1
            elif matrix[w_1][w_2] > target:
                r = mid - 1

        return False


if __name__ == '__main__':
    matrix = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 50]]
    matrix = [[1,1]]
    target = 3
    print(Solution.searchMatrix2(matrix, target))
