from typing import *


class Solution:

    def countKDifference(self, nums: List[int], k: int) -> int:
        nums_length = len(nums)
        nums.sort()
        if nums_length <= 1: return 0
        nums_count = 0
        for i in range(nums_length):
            for j in range(i + 1, nums_length):
                print(nums[j] , nums[i],nums[j] - nums[i],k)
                if nums[j] - nums[i] < k: continue
                if (nums[j] - nums[i]) == k:nums_count = nums_count + 1
                if nums[j] - nums[i] > k: break
        return nums_count


if __name__ == '__main__':
    solution = Solution()
    nums = [1, 2, 2, 1]
    k = 1
    count = solution.countKDifference(nums, k)
    print(count)