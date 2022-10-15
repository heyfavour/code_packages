class Solution:
    def removeInvalidParentheses(self, s: str) -> list[str]:
        left, right = 0, 0
        for i in s:
            if i == "(": left = left + 1
            if i == ")":
                if left > 0:
                    left = left - 1
                else:
                    right = right + 1

        def check(s):
            left = 0
            for i in s:
                if i == "(": left = left + 1
                if i == ")":
                    if left > 0:
                        left = left - 1
                    else:
                        return False
            return left == 0

        ans = []

        def dfs(s, i, l, r):
            if l == r == 0:
                if check(s): ans.append(s)
                return
            for j in range(i, len(s)):
                if j > i and s[j] == s[j - 1]: continue
                if l > 0 and s[j] == "(": dfs(s[:j] + s[j + 1:], j, l - 1, r)
                if r > 0 and s[j] == ")": dfs(s[:j] + s[j + 1:], j, l, r - 1)

        dfs(s, 0, left, right)
        return ans


if __name__ == '__main__':
    solution = Solution()
    s = "()())()"
    print(solution.removeInvalidParentheses(s))
