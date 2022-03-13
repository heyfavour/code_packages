class Solution:
    def canJump(self, nums: list[int]) -> bool:
        max_end, last_end = 0, 0
        for i in range(len(nums)):
            max_end = (nums[i] + i, max_end)
            if i == max_end and nums[i] == 0:
                if max_end < len(nums): return False
        return True


if __name__ == '__main__':
    solution = Solution()
    nums = [2, 3, 1, 1, 4]
    nums = [3, 2, 1, 0, 4]
    print(solution.canJump(nums))
