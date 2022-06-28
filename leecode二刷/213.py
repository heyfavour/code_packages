class Solution:
    def rob(self, nums: list[int]) -> int:
        if len(nums) <= 2: return max(nums)
        n = len(nums)

        def rob(nums):
            if len(nums) <= 2: return max(nums)
            dp = [0] * (n - 1)
            dp[0] = nums[0]
            dp[1] = max(nums[0], nums[1])
            for i in range(2,n - 1):
                dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
            return dp[-1]

        return max(rob(nums[:-1]), rob(nums[1:]))
