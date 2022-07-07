class Solution:
    def calc(self,a,b,opt):
        if opt == "+": return a + b
        if opt == "-": return a - b
        if opt == "*": return a * b

    def diffWaysToCompute(self, expression: str) -> list[int]:
        if len(expression) <= 2: return [int(expression)]
        n = len(expression)
        ans = []
        for i in range(n):
            c = expression[i]
            if not c.isdigit():
                left = self.diffWaysToCompute(expression[:i])
                right = self.diffWaysToCompute(expression[i+1:])
                for a in left:
                    for b in right:
                        ans.append(self.calc(a,b,c))
        return ans