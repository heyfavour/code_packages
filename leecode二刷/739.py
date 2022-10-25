import heapq


class Solution:
    def dailyTemperatures(self, temperatures: list[int]) -> list[int]:
        stack = []
        n = len(temperatures)
        ans = [0] * n
        for i in range(n):
            if i == 0:
                stack.append(i)
            else:
                while stack and temperatures[i] > temperatures[stack[-1]]:
                    index = stack.pop()
                    ans[index] = i - index
                stack.append(i)
        return ans


if __name__ == '__main__':
    solution = Solution()
    temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
    print(solution.dailyTemperatures(temperatures))
