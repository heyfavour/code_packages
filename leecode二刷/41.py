class Solution:
    """
    def firstMissingPositive(self, nums: list[int]) -> int:
        n = len(nums)
        for i in range(n):
            while 1 <= nums[i] <= n:
                if nums[i] == nums[nums[i] - 1]: break
                nums[nums[i] - 1],nums[i]  = nums[i],nums[nums[i] - 1],
        for i in range(n):
            if nums[i] != i + 1: return i + 1
        return n + 1
    """
    def firstMissingPositive(self, nums: list[int]) -> int:
        n = len(nums)
        _dict = {i:i for i in nums if i>0}
        for i in range(1,n+1):
            if _dict.get(i) is None:return i
        return n + 1


if __name__ == '__main__':
    nums = [3,4,-1,1]
    solution = Solution()
    print(solution.firstMissingPositive(nums))
