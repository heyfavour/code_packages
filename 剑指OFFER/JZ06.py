# -*- coding:utf-8 -*-
class Solution:
    @classmethod
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        if len(rotateArray) == 0: return 0
        l, r = 0, len(rotateArray) - 1
        while l <= r:
            mid = (l + r) // 2
            print(mid)
            if rotateArray[mid] < rotateArray[mid + 1] and rotateArray[mid] > rotateArray[mid - 1]:
                return mid
            elif rotateArray[mid] > rotateArray[0]:
                l = mid + 1
            elif rotateArray[mid] < rotateArray[0]:
                r = mid - 1
        return rotateArray[mid]


if __name__ == '__main__':
    print(Solution.minNumberInRotateArray([3,4,5,1,2]))
