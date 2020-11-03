from typing import List


class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        def helper(nums, l, r):
            if l < r:
                mid = (l + r) // 2
                if nums[mid] == target:
                    return mid
                if nums[mid] < nums[mid]
            return -1

        helper(nums, 0, len(nums) - 1)


if __name__ == '__main__':
    nums = [2, 5, 6, 0, 0, 1, 2]
    target = 3
    print(Solution.search(nums, target))
