class Solution(object):
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        sum = 0
        n = len(nums)
        dp = [float("-inf")] * n
        left = 0
        for right in range(n):
            sum = sum + nums[right]
            if sum < target: continue
            while sum >= target:
                dp[right] = left
                sum = sum - nums[left]
                left = left + 1

            left = left - 1
            sum = sum + nums[left]
        ans = min([i - dp[i] + 1 for i in range(n)])
        if ans > n: return 0
        return ans
