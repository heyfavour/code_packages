from typing import *
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        nums_length = len(nums)
        for i in range(nums_length):
            for j in range(i+1,nums_length):
                if nums[i] + nums[j] < target:continue
                if nums[i] + nums[j] == target:return (i,j)
                if nums[i] + nums[j] > target:break

if __name__ == '__main__':
    nums = [2, 7, 11, 15]
    target = 9
    solution = Solution()
    answer = solution.twoSum(nums,target)
    print(answer)