# -*- coding:utf-8 -*-
class Solution:
    @classmethod
    def GetNumberOfK(self, data, k):
        # write code here
        l, r = 0, len(data) - 1
        while l <= r:
            mid = (l + r) // 2
            if data[mid] >= k:
                r = r - 1
            elif data[mid] < k:
                l = l + 1
        lower = l
        l, r = l, len(data) - 1
        while l <= r:
            mid = (l + r) // 2
            if data[mid] > k:
                r = r - 1
            elif data[mid] <= k:
                l = l + 1
        return r-lower + 1

if __name__ == '__main__':
    L, k = [1, 2, 3, 3, 3, 3, 4, 5], 3
    print(Solution.GetNumberOfK(L, k))
