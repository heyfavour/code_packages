from typing import List


class Solution:
    @classmethod
    def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
        map_dict = {}
        res = 0
        for i in range(len(A)):
            for j in range(len(B)):
                map_dict[A[i] + B[j]] = map_dict.get(A[i] + B[j], 0) + 1
        for m in range(len(C)):
            for n in range(len(D)):
                T = map_dict.get(-C[m] - D[n])
                if T is not None: res = res + T

        return res


if __name__ == '__main__':
    A = [-1, -1]
    B = [-1, 1]
    C = [-1, 1]
    D = [1, -1]
    print(Solution.fourSumCount(A, B, C, D))
