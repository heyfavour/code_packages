class Solution:
    def findDuplicate(self, nums: list[int]) -> int:
        slow,fast =nums[0],nums[nums[0]]
        while slow!=fast:
            slow = nums[slow]
            fast = nums[nums[fast]]
        slow = 0
        while slow!=fast:
            slow = nums[slow]
            fast = nums[fast]
        return slow


if __name__ == '__main__':
    solution = Solution()
    nums = [1, 3, 4, 2, 2]
    print(solution.findDuplicate(nums))
