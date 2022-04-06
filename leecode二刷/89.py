class Solution:
    def grayCode(self, n: int) -> list[int]:
        ans = [0] * (1 << n)
        for i in range(1 << n):
            ans[i] = (i >> 1) ^ i
        return ans

if __name__ == '__main__':
    solution = Solution()
    print(solution.grayCode(3))