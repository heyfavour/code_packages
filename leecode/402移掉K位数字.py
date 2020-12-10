class Solution:
    @classmethod
    def removeKdigits(self, num: str, k: int) -> str:
        stack = []
        for i in num:
            while stack and k and stack[-1] > i:
                stack.pop()
                k = k - 1
            stack.append(i)
        if k: stack = stack[:-k]
        return "".join(stack).lstrip("0") or "0"


if __name__ == '__main__':
    num, k = "1432219", 3
    print(Solution.removeKdigits(num, k))
