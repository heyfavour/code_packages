from typing import List


class Solution:
    @classmethod
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        result = [[],]

        def helper(L, poi, length):
            if len(L) == length and L not in result:
                result.append(L[:])
                return
            for i in range(poi, len(nums)):
                if i>poi and nums[i] == nums[i-1]:continue
                L.append(nums[i])
                helper(L, i + 1, length)
                L.pop()

        for i in range(1, len(nums) + 1):
            helper([], 0, i)

        return result
if __name__ == '__main__':
    L = [1, 2, 2]
    print(Solution.subsetsWithDup(L))
