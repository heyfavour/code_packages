from typing import List


class Solution:
    #超时
    def maxSum1(self, nums1: List[int], nums2: List[int]) -> int:
        self.sum_list = 0

        def switch(nums1, nums2):
            if nums2 != [] and nums1 != []:
                for i, v in enumerate(nums1):
                    if v in nums2:
                        si = nums2.index(v)
                        max_sum = max(sum(nums1[:i + 1]), sum(nums2[:si + 1]))
                        self.sum_list = self.sum_list + max_sum
                        switch(nums1[i + 1:], nums2[si + 1:])
                        return
            self.sum_list = self.sum_list + max(sum(nums1), sum(nums2))

        switch(nums1, nums2)

        return (self.sum_list) % (10 ** 9 + 7)
    #双指针
    def maxSum2(self, nums1: List[int], nums2: List[int]) -> int:
        self.sum_list = 0
        sum1 = 0
        sum2 = 0
        i = 0
        j = 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] < nums2[j]:
                sum1 = sum1 + nums1[i]
                i = i + 1
            elif nums1[i] > nums2[j]:
                sum2 = sum2 + nums2[j]
                j = j + 1
            else:
                self.sum_list = self.sum_list + max(sum1, sum2) + nums1[i]
                sum1 = sum2 = 0
                i = i+1
                j = j+1
        sum1 =sum1+sum(nums1[i:])
        sum2 =sum2+sum(nums2[j:])
        self.sum_list = self.sum_list + max(sum1, sum2)

        return (self.sum_list) % (10 ** 9 + 7)


if __name__ == '__main__':
    nums1 = [2, 4, 5, 8, 9,10,11]
    nums2 = [4, 6, 8, 9,12,13]

    s = Solution()
    print(s.maxSum1(nums1, nums2))
    print(s.maxSum2(nums1, nums2))
