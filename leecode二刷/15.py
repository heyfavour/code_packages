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

    def threeSum(self, nums: list[int]) -> list[list[int]]:

        nums_len = len(nums)
        if nums_len < 3: return []
        nums.sort()
        ans = []
        for i in range(nums_len - 2):
            if nums[i] > 0: break
            if i >= 1 and nums[i] == nums[i - 1]: continue
            L = i + 1
            R = nums_len - 1
            while L < R:
                if nums[i] + nums[L] + nums[R] == 0:
                    ans.append([nums[i], nums[L], nums[R]])
                    while L < R and nums[L] == nums[L + 1]: L = L + 1
                    while L < R and nums[R] == nums[R - 1]: R = R - 1
                    L = L + 1
                    R = R - 1
                elif nums[i] + nums[L] + nums[R] < 0:
                    L = L + 1
                elif nums[i] + nums[L] + nums[R] > 0:
                    R = R - 1
        return ans


if __name__ == '__main__':
    solution = Solution()
    nums = [-1, 0, 1, 2, -1, -4]
    print(solution.threeSum(nums))
