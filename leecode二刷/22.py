class Solution:
    def generateParenthesis(self, n: int) -> list[str]:
        ans = []
        def add_arenthesis(left, right, string):
            print(left,right,string)
            if left == 0 and right == 0: ans.append(string)
            if left > 0: add_arenthesis(left - 1, right, string+"(")
            if right > 0 and right > left: add_arenthesis(left , right - 1, string+")")

        add_arenthesis(n, n, "")
        return ans


if __name__ == '__main__':
    n = 3
    solution = Solution()
    print(solution.generateParenthesis(n))
