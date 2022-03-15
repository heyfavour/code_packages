import time


class Solution:
    def generateMatrix(self, n: int) -> list[list[int]]:
        l, r, t, b = 0, n - 1, 0, n - 1
        num, end = 1, n * n
        martix = [[0] * n for i in range(n)]
        while num <= end:
            if l == r == t == b:
                martix[l][t] = num
                num = num + 1
            for i in range(l, r):
                martix[t][i] = num
                num = num + 1
            for i in range(t, b):
                martix[i][r] = num
                num = num + 1
            for i in range(r, l, -1):
                martix[b][i] = num
                num = num + 1
            for i in range(b, t, -1):
                martix[i][l] = num
                num = num + 1
            l, r, t, b = l + 1, r - 1, t + 1, b - 1
        return martix


if __name__ == '__main__':
    solution = Solution()
    print(solution.generateMatrix(5))
