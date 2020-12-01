from typing import List


class Solution:
    @classmethod
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        n = len(nums)
        min_delta = nums[0] + nums[1] + nums[2] - target
        print(min_delta)
        for i in range(n):
            if i > 0 and nums[i] == nums[i - 1]: continue
            L = i + 1
            R = n - 1
            while L < R:
                delta = nums[i] + nums[L] + nums[R] - target
                if abs(delta) < abs(min_delta):
                    min_delta = delta
                if delta == 0:
                    return target
                elif delta < 0:
                    L = L + 1
                elif delta > 0:
                    R = R - 1
        return min_delta + target


if __name__ == '__main__':
    nums = [-1, 2, 1, -4]
    nums = [0, 0, 0]
    print(Solution.threeSumClosest(nums, 1))
