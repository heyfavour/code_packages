from typing import List


class Solution:
    @classmethod
    def validMountainArray(self, A: List[int]) -> bool:
        down = False
        if len(A) <= 2: return False
        if A[0] >= A[1]: return False
        for i in range(len(A) - 1):
            print(A[i], A[i + 1])
            if down == False:
                if A[i] > A[i + 1]:
                    down = True
                elif A[i] == A[i + 1]:
                    return False
            elif down == True:
                if A[i] <= A[i + 1]:
                    return False
        return True


if __name__ == '__main__':
    L = [0, 1, 2, 4, 2, 1]
    L = [3,5,5]
    print(Solution.validMountainArray(L))
