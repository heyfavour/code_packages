class Solution:
    def threeSumClosest(self, nums: list[int], target: int) -> int:
        nums_len = len(nums)
        nums.sort()
        ans = 10 ** 5
        for i in range(nums_len - 2):
            L = i + 1
            R = nums_len - 1
            while L < R:
                three_sum = nums[i] + nums[L] + nums[R]
                if three_sum == target:
                    return three_sum
                elif three_sum < target:
                    if abs(three_sum - target) < abs(ans - target) :ans = three_sum
                    L = L + 1
                elif three_sum > target:
                    if abs(three_sum - target) < abs(ans - target): ans = three_sum
                    R = R - 1
        return ans


if __name__ == '__main__':
    nums = [-1, 2, 1, -4]
    solution = Solution()
    target = 1
    print(solution.threeSumClosest(nums,target))
