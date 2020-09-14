from typing import List
from copy import deepcopy
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result = []
        n = len(nums)
        def back(nums,path):
            if len(path) == n:
                result.append(path)
                return
            for i in range(len(nums)):
                new_path = deepcopy(path)
                new_path.append(nums[i])
                new_nums = deepcopy(nums)
                new_nums.pop(i)
                back(new_nums,new_path)
        back(nums,[])
        return result

if __name__ == '__main__':
    s = Solution()
    l = [1,2,3]
    print(s.permute(l))
