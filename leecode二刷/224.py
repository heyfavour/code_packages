class Solution:
    def calculate(self, s: str) -> int:
        sign, sign_stack = 1, [1, ]
        ans = 0

        i, n = 0, len(s)
        while i < n:
            if s[i].isdigit():
                num = 0
                while i < n and s[i].isdigit():
                    num = num * 10 + int(s[i])
                    i = i+1
                ans = ans + sign * num
            else:
                if s[i] == "+":
                    sign = sign_stack[-1]
                elif s[i] == "-":
                    sign = -1 * sign_stack[-1]
                elif s[i] == "(":
                    sign_stack.append(sign)
                elif s[i] == ")":
                    sign_stack.pop()
                i = i + 1
        return ans


if __name__ == '__main__':
    solution = Solution()
    s = "(1+(4+5+2)-3)+(6+8)"
    print(solution.calculate(s))
