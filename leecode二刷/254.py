import math


class Solution:
    def getFactors(self, n: int) -> list[list[int]]:
        def dfs(n,factor):
            ans = []
            for i in range(factor, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    ans.append([i, n // i])
                    for j in dfs(n//i,i):#n/2  下次开始的数字
                        ans.append([i]+j)
            return ans
        return dfs(n,2)


if __name__ == '__main__':
    solution = Solution()
    print(solution.getFactors(8))
