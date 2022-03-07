class Solution:
    """
    #模拟法
    def longestValidParentheses(self, s: str) -> int:
        n = len(s)
        nums = [0] * n
        stack = []
        for i in range(n):
            if s[i] == "(":
                stack.append(i)
            else:
                if stack:
                    stack.pop()
                else:
                    nums[i] = 1
        for i in stack: nums[i] = 1
        print(nums)
        _len = 0
        _max = 0
        for i in nums:
            if i:
                _len = 0
            else:
                _len = _len + 1
                _max = max((_max, _len))
        return _max
        """

    # 动态规划
    def longestValidParentheses(self, s: str) -> int:
        # - i-1-dp[i-1] - - - i-1 i
        n = len(s)
        if n <= 1: return 0
        dp = [0] * n
        for i in range(n):
            if s[i] == ")" and i - dp[i - 1] - 1 >= 0 and s[i - dp[i - 1] - 1] == "(":
                pre = dp[i - dp[i - 1] - 2] if i - dp[i - 1] - 2 > 0 else 0
                dp[i] = pre + dp[i - 1] + 2
        return max(dp)


if __name__ == '__main__':
    s = "(()"
    solution = Solution()
    print(solution.longestValidParentheses(s))
