class Solution:
    def goodDaysToRobBank(self, security: list[int], time: int) -> list[int]:
        n = len(security)
        up = [0] * n
        down = [0] * n
        for i in range(0, n):
            if i > 0 and security[i - 1] >= security[i]:
                up[i] = up[i - 1] + 1

        for i in range(n-2,-1,-1):
            if security[i]<=security[i+1]:
                down[i] = down[i+1]+1
        ans = []
        for i in range(time, n - time):
            if up[i]>=time and down[i]>=time:ans.append(i)
        return ans


if __name__ == '__main__':
    solution = Solution()
    security = [5, 3, 3, 3, 5, 6, 2]
    time = 2
    print(solution.goodDaysToRobBank(security, time))
