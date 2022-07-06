class Solution:
    def calculate(self, s: str) -> int:
        s = ("("+s.replace(" ", "")+")").replace("(-", "(0-")
        priorty_dict = {"+": 1, "-": 1, "*": 2, "/": 2}
        num_stack = []
        opt_stack = []

        def calc(num_stack, opt_stack):
            print(num_stack,opt_stack)
            b, a, opt = num_stack.pop(), num_stack.pop() if num_stack else 0, opt_stack.pop()
            if opt == "+":
                num_stack.append(a + b)
            elif opt == "-":
                num_stack.append(a - b)
            elif opt == "*":
                num_stack.append(a * b)
            elif opt == "/":
                num_stack.append(a / b)

        i, n = 0, len(s)
        while i < n:
            c = s[i]
            if c.isdigit():
                num = 0
                while i < n and s[i].isdigit():
                    num = num * 10 + int(s[i])
                    i = i + 1
                num_stack.append(num)
            else:
                if c == "(":
                    opt_stack.append(c)
                elif c == ")":  # 清算(
                    while opt_stack and opt_stack[-1] != "(":
                        calc(num_stack, opt_stack)
                    opt_stack.pop()
                else:  # + - * / 2-x*2/2
                    #将所有opt_stack内优先级比当前高的处理好
                    while opt_stack and opt_stack[-1] != "(" and priorty_dict[opt_stack[-1]] >= priorty_dict[c]:
                        calc(num_stack, opt_stack)
                    opt_stack.append(c)
                i = i + 1

        return num_stack[0]


if __name__ == '__main__':
    s = "-2-2*2/2"
    solution = Solution()
    print(solution.calculate(s))
