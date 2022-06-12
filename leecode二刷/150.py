class Solution:
    def evalRPN(self, tokens: list[str]) -> int:
        queue = []
        for i in tokens:
            print(i,queue)
            if i not in ("+","-","*","/"):
                queue.append(int(i))
            else:
                a = queue.pop()
                b = queue.pop()
                if i == "+":queue.append(b+a)
                elif i == "/":queue.append(int(b/a))
                elif i == "*":queue.append(b*a)
                elif i == "-":queue.append(b-a)
        return queue[0]




if __name__ == '__main__':
    solution = Solution()
    tokens = ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
    print(solution.evalRPN(tokens))