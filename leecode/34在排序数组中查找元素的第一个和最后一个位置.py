from typing import List


class Solution:
    @classmethod
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if nums[0] > target or nums[-1] < target: return [-1, -1]
        l, r = 0, len(nums) - 1

        while l < r:
            mid = (l + r) // 2
            if nums[mid] >= target:
                r = r - 1
            elif nums[mid] < target:
                l = l + 1
        print(l, r)
        l, r = 0, len(nums) - 1
        while l < r:
            mid = (l + r) // 2
            if nums[mid] <= target:
                l = l + 1
            elif nums[mid] < target:
                r = r - 1

        print(l, r)


if __name__ == '__main__':
    nums = [5, 7, 7, 8, 8, 8, 8, 10]
    target = 6
    print(Solution.searchRange(nums, target))
