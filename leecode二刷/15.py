class Solution:
    """
    超时
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        nums_len = len(nums)
        ans = []
        for i in range(nums_len-2):
            for j in range(i+1,nums_len-1):
                for m in range(j+1,nums_len):
                    if nums[i]+nums[j]+nums[m] == 0 and sorted([nums[i],nums[j],nums[m]]) not in ans:
                        ans.append(sorted([nums[i],nums[j],nums[m]]))
        return ans
    """
if __name__ == '__main__':
    solution = Solution()
    nums = [-1, 0, 1, 2, -1, -4]
    print(solution.threeSum(nums))