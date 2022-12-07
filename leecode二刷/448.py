class Solution:
    def findDisappearedNumbers(self, nums: list[int]) -> list[int]:
        for i in range(len(nums)):
            while nums[i]!=nums[nums[i]-1]:
                nums[nums[i]-1],nums[i] = nums[i],nums[nums[i]-1]
        return [i+1 for i,v in enumerate(nums) if i+1!=v]