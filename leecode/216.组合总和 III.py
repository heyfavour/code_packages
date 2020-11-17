from typing import List


class Solution:
    @classmethod
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        result = []

        def helper(L, s):
            if len(L) == k and sum(L) == n:
                result.append(L[:])
                return
            for i in range(s, 10):
                if len(L)>k:break
                L.append(i)
                helper(L, i + 1)
                L.pop()

        helper([], 1)
        return result


if __name__ == '__main__':
    k = 3
    n = 9
    print(Solution.combinationSum3(k, n))
