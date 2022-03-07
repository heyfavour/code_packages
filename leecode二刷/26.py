class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        slow = 1
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1]:
                nums[slow] = nums[i]
                slow = slow + 1
        return slow


if __name__ == '__main__':
    solution = Solution()
    nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
    print(solution.removeDuplicates(nums))
    print(nums)
