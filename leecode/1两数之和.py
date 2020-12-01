from typing import List


class Solution:
    @classmethod
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        nums_dict = {}
        for i in range(len(nums)):
            if nums_dict.get(target - nums[i]) is not None:
                return [nums_dict.get(target - nums[i]), i]
            nums_dict[nums[i]] = i


if __name__ == '__main__':
    nums = [2, 7, 11, 15]
    target = 9
    print(Solution.twoSum(nums, target))
