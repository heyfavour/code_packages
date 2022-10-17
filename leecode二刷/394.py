class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        k = 0
        res = ""
        for char in s:
            if char.isdigit():
                k = k * 10 + int(char)
            elif char == "[":
                stack.append((res, k or 1))
                res, k = "", 0
            elif char == "]":
                stack_res, num = stack.pop()
                res = stack_res + res * num
            else:
                res = res + char
        return res


if __name__ == '__main__':
    solution = Solution()
    print(solution.decodeString("3[a2[c]]"))
