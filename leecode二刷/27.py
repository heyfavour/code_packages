class Solution:
    def removeElement(self, nums: list[int], val: int) -> int:
        slow = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[slow] = nums[i]
                slow = slow + 1
        return slow


if __name__ == '__main__':
    solution = Solution()
    nums = [1, 1, 2]
    val = 2
    print(solution.removeElement(nums, val))
