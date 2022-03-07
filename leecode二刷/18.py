class Solution:
    def fourSum(self, nums: list[int], target: int) -> list[list[int]]:
        nums.sort()
        nums_len = len(nums)
        ans = []
        for i in range(nums_len - 3):
            if i > 0 and nums[i - 1] == nums[i]: continue
            for j in range(i + 1, nums_len - 2):
                if j > i + 1 and nums[j - 1] == nums[j]: continue
                L = j + 1
                R = nums_len - 1
                while L < R:
                    four_sum = nums[i] + nums[j] + nums[L] + nums[R]
                    if four_sum == target:
                        ans.append([nums[i], nums[j], nums[L], nums[R]])
                        while L < R and nums[L] == nums[L + 1]: L = L + 1
                        while L < R and nums[R] == nums[R - 1]: R = R - 1
                        L = L + 1
                        R = R - 1
                    elif four_sum < target:
                        L = L + 1
                    elif four_sum > target:
                        R = R - 1
        return ans


if __name__ == '__main__':
    nums = [1, 0, -1, 0, -2, 2]
    nums = [2, 2, 2, 2, 2]
    target = 8
    solution = Solution()
    print(solution.fourSum(nums, target))
