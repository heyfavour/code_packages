class Solution:
    def findMedianSortedArrays(self, nums1: list[int], nums2: list[int]) -> float:
        def getKthElement(k):
            index1, index2 = 0, 0
            while True:
                if index1 == m:return nums2[index2 + k - 1]
                if index2 == n:return nums1[index1 + k - 1]
                if k == 1:return min(nums1[index1], nums2[index2])
                newIndex1 = min(index1 + k // 2 - 1, m - 1)
                newIndex2 = min(index2 + k // 2 - 1, n - 1)
                pivot1, pivot2 = nums1[newIndex1], nums2[newIndex2]
                if pivot1 <= pivot2:
                    k -= newIndex1 - index1 + 1
                    index1 = newIndex1 + 1
                else:
                    k -= newIndex2 - index2 + 1
                    index2 = newIndex2 + 1

        m, n = len(nums1), len(nums2)
        all = m + n

        if all % 2 == 1:return getKthElement((m + n + 1) // 2) #奇数 只有一个 寻找 (m+ n + 1)//2
        return (getKthElement(all // 2) + getKthElement(all // 2 + 1)) / 2



if __name__ == '__main__':
    nums1 = [1, 3]
    nums2 = [2]
    solution = Solution()
    median = solution.findMedianSortedArrays(nums1,nums2)
    print(median)