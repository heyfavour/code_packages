class Solution:
    def rotate(self, nums: list[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        def swap(l, r):
            while l < r:
                nums[l], nums[r] = nums[r], nums[l]
                l = l + 1
                r = r - 1

        n = len(nums)
        swap(0, n - 1)
        k = k % n
        swap(0, k - 1)
        swap(k, n - 1)