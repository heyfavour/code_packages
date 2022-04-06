class Solution:
    def merge(self, nums1: list[int], m: int, nums2: list[int], n: int) -> None:
        i = m
        j = n
        while i and j:
            if nums2[j - 1] > nums1[i - 1]:
                nums1[i + j - 1] = nums2[j - 1]
                j = j - 1
            else:
                nums1[i + j - 1] = nums1[i - 1]
                i = i - 1
        if j: nums1[:j] = nums2[:j]
        return nums1


if __name__ == '__main__':
    nums1 = [1, 2, 3, 0, 0, 0]
    nums2 = [2, 5, 6]
    m = 3
    n = 3
    solution = Solution()
    print(solution.merge(nums1, m, nums2, n))
