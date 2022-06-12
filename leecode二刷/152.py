class Solution:
    def maxProduct(self, nums: list[int]) -> int:
        n = len(nums)
        dp_max = [float("-inf")]*n
        dp_min = [float("inf")]*n
        for i in range(n):
            if i == 0:
                dp_max[0] = nums[0]
                dp_min[0] = nums[0]
            else:
                dp_max[i] = max([dp_max[i-1]*nums[i],dp_min[i-1]*nums[i],nums[i]])
                dp_min[i] = max([dp_max[i-1]*nums[i],dp_min[i-1]*nums[i],nums[i]])
        return max(dp_max)


if __name__ == '__main__':
    solution = Solution()
    nums = [-2,3,-4]
    print(solution.maxProduct(nums))
