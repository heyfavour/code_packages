from typing import List
import copy


class Solution:
    @classmethod
    def subsets(self, nums: List[int]) -> List[List[int]]:
        result = []

        def helper(L, postition, length):
            if len(L) == length:
                result.append(L)
                return
            for i in range(postition, len(nums)):
                NL = copy.deepcopy(L)
                NL.append(nums[i])
                helper(NL, i + 1, length)

        for i in range(1, len(nums) + 1):
            helper([], 0, i)
        return result


if __name__ == '__main__':
    nums = [1, 2, 3]
    print(Solution.subsets(nums))
