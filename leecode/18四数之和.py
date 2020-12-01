from typing import List


class Solution:
    @classmethod
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        res = []
        n = len(nums)
        print(nums)
        for i in range(n):
            if i > 0 and nums[i] == nums[i - 1]: continue
            for j in range(i + 1, n):
                if j > i + 1 and nums[j] == nums[j - 1]: continue
                L = j + 1
                R = n - 1
                while L < R:
                    if nums[i] + nums[j] + nums[L] + nums[R] == target:
                        res.append([nums[i], nums[j], nums[L], nums[R]])
                        while L < R and nums[L] == nums[L + 1]: L = L + 1
                        while L < R and nums[R] == nums[R - 1]: R = R - 1
                        L = L - 1
                        R = R - 1
                    elif nums[i] + nums[j] + nums[L] + nums[R] < target:
                        L = L + 1
                    elif nums[i] + nums[j] + nums[L] + nums[R] > target:
                        R = R - 1
        return res


if __name__ == '__main__':
    nums = [1, 0, -1, 0, -2, 2]
    nums = [0,0,0,0]
    nums = [1,-2,-5,-4,-3,3,3,5]
    target = -11
    print(Solution.fourSum(nums, target))
