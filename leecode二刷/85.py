class Solution:
    def maximalRectangle(self, matrix: list[list[str]]) -> int:
        m, n = len(matrix), len(matrix[0])
        heights = [[0] * n for _ in range(m)]
        heights[0] = [int(i) for i in matrix[0]]
        ans = []

        def deal_one_row(nums):
            stack = []
            nums.append(0)
            for i in range(n+1):
                while stack and stack[-1][1] > nums[i]:
                    H = stack.pop()[1]
                    while stack and stack[-1][1] == H: stack.pop()
                    if stack:W = i - stack[-1][0] - 1
                    else:W = i
                    ans.append(W * H)
                stack.append((i, nums[i]))

        for i in range(1, m):
            for j in range(n):
                if matrix[i][j] == "1": heights[i][j] = heights[i - 1][j] + 1
            deal_one_row(heights[i])
        return max(ans)

if __name__ == '__main__':
    solution = Solution()
    matrix = [["1", "0", "1", "0", "0"], ["1", "0", "1", "1", "1"], ["1", "1", "1", "1", "1"],
              ["1", "0", "0", "1", "0"]]
    print(solution.maximalRectangle(matrix))
