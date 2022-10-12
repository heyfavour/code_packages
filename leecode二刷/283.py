class Solution:
    def moveZeroes(self, nums: list[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if not nums: return nums
        l, r = 0, 0
        while r < len(nums):
            if nums[r] != 0:
                nums[l], nums[r] = nums[r], nums[l]
                l = l + 1
            r = r + 1
        return nums


if __name__ == '__main__':
    nums = [0, 1, 0, 3, 12]
    solution = Solution()
    print(solution.moveZeroes(nums))
