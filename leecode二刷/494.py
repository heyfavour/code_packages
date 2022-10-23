class Solution:
    def findTargetSumWays(self, nums: list[int], target: int) -> int:
        # n = len(nums)
        # ans = 0
        # def dfs(index,sum):
        #     if index == n:
        #         nonlocal ans
        #         if sum == target:ans=ans+1
        #         return
        #     sum = sum + dfs(index+1,sum + nums[index])
        #     dfs(index+1,sum - nums[index])
        # dfs(0,0)
        # return ans
        diff = sum(nums) - target
        if diff < 0 or diff % 2 == 1: return 0
        nums.insert(0, 0)
        c = int(diff / 2 + 1)
        dp = [[0] * c for _ in range(len(nums))]
        dp[0][0] = 1
        for i in range(1, len(nums)):
            for j in range(c):
                if j >= nums[i]:
                    dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i]]
                else:
                    dp[i][j] = dp[i - 1][j]
        return dp[-1][-1]


if __name__ == '__main__':
    solution = Solution()
    nums = [1, 1, 1, 1, 1]
    target = 3
    print(solution.findTargetSumWays(nums, target))
