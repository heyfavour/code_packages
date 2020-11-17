# -*- coding:utf-8 -*-
class Solution:
    # array 二维列表
    @classmethod
    def Find(self, target, array):
        # write code here
        row, col = 0, len(array[0]) - 1
        while 0 <= row <= len(array) - 1 and 0 <= col <= len(array[0]) - 1:
            if array[row][col] == target:
                return True
            elif target > array[row][col]:
                row = row + 1
            elif target < array[row][col]:
                col = col - 1
        return False


if __name__ == '__main__':
    target, array = 7, [[1, 2, 8, 9], [2, 4, 9, 12], [4, 7, 10, 13], [6, 8, 11, 15]]
    print(Solution.Find(target, array))
