class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        i = 0
        while i <= len(nums) - 3:
            if nums[i] == nums[i + 2]:
                nums.pop(i + 2)
            else:
                i = i + 1
        return len(nums)


if __name__ == '__main__':
    nums = [0, 0, 1, 1, 1, 1, 2, 3, 3]
    solution = Solution()
    print(solution.removeDuplicates(nums))
