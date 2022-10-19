class Solution:
    def canPartition(self, nums: list[int]) -> bool:
        _sum = sum(nums)
        if _sum %2 == 1:return False
        c = int(_sum/2)
        nums.insert(0,0)
        dp = [[0]*(c+1) for _ in range(len(nums))]
        for i in range(len(nums)):
            for j in range(1,c+1):
                if j>=nums[i]:
                    dp[i][j] = max(dp[i-1][j],dp[i-1][j-nums[i]]+nums[i])
                else:
                    dp[i][j] = dp[i-1][j]
                if dp[i][j] == c:return True
        return False
        """
        nums_sum = sum(nums)
        if nums_sum %2 == 1:return False
        half_sum = int(nums_sum/2)
        nums.insert(0,0)
        dp = [0]*(half_sum + 1)
        for num in nums:
            for j in range(half_sum,num-1,-1):
                dp[j] = max(dp[j],dp[j-num]+num)
                if dp[j] == half_sum:return True
        return False
        """



if __name__ == '__main__':
    solution = Solution()
    nums = [1, 5, 11, 5]
    print(solution.canPartition(nums))