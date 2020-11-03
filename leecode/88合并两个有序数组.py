from typing import List


class Solution:
    @classmethod
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        i = len(nums1) - 1
        while  m-1>=0 and n-1>=0:
            if nums1[m - 1] < nums2[n - 1]:
                nums1[i] = nums2[n - 1]
                n = n - 1
            else:
                nums1[i] = nums1[m - 1]
                m = m - 1
            i = i - 1
        if n-1>=0:nums1[0:n] = nums2[0:n]
        return nums1


if __name__ == '__main__':
    nums1 = [3, 3, 3, 0, 0, 0]
    m = 3
    nums2 = [2, 2, 2]
    n = 3
    print(Solution.merge(nums1, m, nums2, n))
