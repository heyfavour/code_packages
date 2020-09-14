from typing import List
import copy

class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        result = []
        def back(nums, path):
            if len(path) == n:
                result.append(path)
            for i in range(len(nums)):
                if nums[i] == nums[i-1] and i>0:continue
                new_nums = copy.deepcopy(nums)
                new_nums.pop(i)
                new_path =copy.deepcopy(path)
                new_path.append(nums[i])
                back(new_nums, new_path)
        back(nums, [])
        return  result


if __name__ == '__main__':
    s = Solution()
    l = [1, 1, 2]
    print(s.permuteUnique(l))
