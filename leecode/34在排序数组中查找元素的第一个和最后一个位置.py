from typing import List


class Solution:
    @classmethod
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if nums[0] > target or nums[-1] < target: return [-1, -1]
        l, r = 0, len(nums) - 1

        while l <= r:
            mid = (l + r) // 2
            if nums[mid] >= target:
                r = mid - 1
            elif nums[mid] < target:
                l = mid + 1
        L = l
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] <= target:
                l = mid + 1
            elif nums[mid] > target:
                r = mid - 1
        return [L, r]


if __name__ == '__main__':
    nums = [5, 7, 7, 8, 8, 10]
    target = 8
    print(Solution.searchRange(nums, target))
