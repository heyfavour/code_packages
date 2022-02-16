class Solution:
    def findMedianSortedArrays(self, nums1: list[int], nums2: list[int]) -> float:
        def get_mid_by_k(k):
            index_1, index_2 = 0, 0
            while True:
                if index_1 == m: return nums2[index_2 + k - 1]
                if index_2 == n: return nums1[index_1 + k - 1]
                if k == 1: return min(nums1[index_1], nums2[index_2])
                tmp_index_1 = min(index_1 + k // 2 - 1, m - 1)
                tmp_index_2 = min(index_2 + k // 2 - 1, n - 1)
                if nums1[tmp_index_1] <= nums2[tmp_index_2]:
                    k = k - (tmp_index_1 - index_1 + 1)
                    index_1 = tmp_index_1 + 1
                else:
                    k = k - (tmp_index_2 - index_2 + 1)
                    index_2 = tmp_index_2 + 1

        m, n = len(nums1), len(nums2)
        if (m + n) % 2 == 1:
            return get_mid_by_k((m + n + 1) // 2)
        else:
            return (get_mid_by_k((m + n) // 2) + get_mid_by_k((m + n) // 2 + 1)) / 2


if __name__ == '__main__':
    nums1 = [1, 3, 5, 7, 9]
    nums2 = [2, 4, 6, 8, 10]
    solution = Solution()
    median = solution.findMedianSortedArrays(nums1, nums2)
    print(median)
