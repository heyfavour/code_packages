from typing import List


class Solution:
    @classmethod
    def containsDuplicate(self, nums: List[int]) -> bool:
        repeat_dict = {}
        for i in nums:
            repeat_dict[i] = repeat_dict.get(i,0) + 1
            if repeat_dict[i]>1:return True
        return False


if __name__ == '__main__':
    L = [1, 1, 1, 3, 3, 4, 3, 2, 4, 2]
    print(Solution.containsDuplicate(L))
