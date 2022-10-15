class Solution:
    def lengthOfLIS(self, nums: list[int]) -> int:
        if not nums:return 0
        dp = [1]*len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[i]>nums[j]:
                    dp[i] = max(dp[i],dp[j]+1)
        return max(dp)

if __name__ == '__main__':
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    solution = Solution()
    print(solution.lengthOfLIS(nums))