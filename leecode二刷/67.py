class Solution:
    def addBinary(self, a: str, b: str) -> str:
        a,b= list(a),list(b)
        m, n = len(a), len(b)
        _len = max((m, n))
        string = ""
        add = 0
        for i in range(_len - 1, -1, -1):
            a_num = int(a.pop()) if a else 0
            b_num = int(b.pop()) if b else 0
            sum = a_num + b_num + add
            add, num = (sum) // 2, sum % 2
            string = str(num) + string
        if add: string = "1" + string
        return string


if __name__ == '__main__':
    a = "1010"
    b = "1011"
    solution = Solution()
    print(solution.addBinary(a, b))
