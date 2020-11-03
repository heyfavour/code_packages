from typing import List


class Solution:
    @classmethod
    def combine(self, n: int, k: int) -> List[List[int]]:
        result = []

        def helper(part, nums):
            if len(part) == k:
                result.append(part[:])
                return
            for i in range(nums+1, n + 1):
                if part!=[] and i <= part[-1]:continue
                part.append(i)
                nums = nums + 1
                helper(part, nums)
                nums = nums - 1
                part.pop()

        helper([], 0)
        return result


if __name__ == '__main__':
    print(Solution.combine(4, 2))
