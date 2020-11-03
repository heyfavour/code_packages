from typing import List


class Solution:
    @classmethod
    def removeDuplicates(self, nums: List[int]) -> int:
        i, j = 0, 0
        while j < len(nums):
            print(i,j,nums)
            if nums[i] == nums[j] and j - i < 2:
                j = j + 1
            elif nums[i] == nums[j] and j - i >= 2:
                nums.pop(j)
            else:
                i = i + 1


if __name__ == '__main__':
    L = [0, 0, 1, 1, 1, 1, 2, 3, 3]
    print(Solution.removeDuplicates(L))
