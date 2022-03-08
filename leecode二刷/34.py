from typing import List


class Solution:
    @classmethod
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums or nums[0] > target or nums[-1] < target: return [-1, -1]
        n = len(nums)
        l, r = 0, n - 1
        while l <= r:
            mid = (l + r) >> 1
            if nums[mid] == target:
                r = mid - 1
            elif nums[mid] > target:
                r = mid - 1
            elif nums[mid] < target:
                l = mid + 1
        if nums[l] != target: return [-1, -1]
        L = l
        l, r = 0, n - 1
        while l <= r:
            mid = (l + r) >> 1
            if nums[mid] == target:
                l = mid + 1
            elif nums[mid] < target:
                l = mid + 1
            elif nums[mid] > target:
                r = mid - 1
        R = r
        return [L, R]

if __name__ == '__main__':
    nums = [5, 7, 7, 8, 8, 10]
    target = 8
    print(Solution.searchRange(nums, target))
