from typing import List
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp = [0]*len(nums)
        dp[0] = nums[0]
        for i in range(1,len(nums)):
            if dp[i-1]<0:
                dp[i] = nums[i]
            else:
                dp[i] = dp[i-1] + nums[i]
        return max(dp)


if __name__ == '__main__':
    s = Solution()
    l = [2,3,-2,4,1,3]
    print(s.maxSubArray(l))
