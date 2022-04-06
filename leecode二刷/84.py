class Solution:
    # 暴力法
    """
    def largestRectangleArea(self, heights: list[int]) -> int:
        n = len(heights)
        ans = []
        for i in range(n):
            left_i = i
            right_i = i
            while left_i - 1 >= 0 and heights[left_i - 1] >= heights[i]:
                left_i = left_i - 1
            while right_i + 1 <= n - 1 and heights[right_i + 1] >= heights[i]:
                right_i = right_i + 1
            area = (right_i-left_i+1)*heights[i]
            ans.append(area)
        return max(ans)
    """
    """
    # 单调栈
    def largestRectangleArea(self, heights: list[int]) -> int:
        n = len(heights)
        ans = []
        stack = []
        for i in range(n):
            while stack and heights[i] < stack[-1][1]:  # 遇到 递减时12333[2]
                H = stack.pop()[-1]  # [3]#3的数据可以出了
                while stack and H == stack[-1][1]:  # 剔除12 [3]
                    stack.pop()
                W = i - ((stack[-1][0] + 1) if stack else 0)
                ans.append(W * H)
            stack.append((i, heights[i]))  # [idx,H]
        print(stack)
        while stack:
            H = stack[-1][1]
            while stack and stack[-1][1] == H:
                stack.pop()
            W = n - ((stack[-1][1] + 1) if stack else 0)
            ans.append(W * H)
        return ans
        """

    # 单调栈 + 哨兵
    def largestRectangleArea(self, heights: list[int]) -> int:
        heights = heights + [0]
        n = len(heights)
        ans, stack = [], []
        for i in range(n):
            while stack and heights[i] < stack[-1][1]:
                H = stack.pop()[1]
                while stack and stack[-1][1] == H:
                    stack.pop()
                if stack:W = i - stack[-1][0] - 1
                else:W = i
                ans.append(W * H)
            stack.append((i, heights[i]))
        return max(ans)


if __name__ == '__main__':
    heights = [2, 1, 5, 6, 2, 3]
    solution = Solution()
    print(solution.largestRectangleArea(heights))
