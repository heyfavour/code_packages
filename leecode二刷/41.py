class Solution:
    def firstMissingPositive(self, nums: list[int]) -> int:
        n = len(nums)
        for i in range(n):
            while 1 <= nums[i] <= n and nums[i] != nums[nums[i] - 1]:
                nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1],
        for i in range(n):
            if nums[i] != i + 1: return i + 1
        return n + 1


if __name__ == '__main__':
    nums = [3, 4, -1, 1]
    solution = Solution()
    print(solution.firstMissingPositive(nums))
