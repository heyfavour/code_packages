class Solution:
    def fractionToDecimal(self, numerator: int, denominator: int) -> str:
        if numerator % denominator == 0: return str(numerator // denominator)

        ans = []
        if numerator < 0 and denominator < 0: ans.append("-")
        numerator = abs(numerator)
        denominator = abs(denominator)

        ans.append(str(numerator // denominator))
        ans.append(".")

        remainder_map = {}
        remainder = numerator % denominator

        while remainder and remainder not in remainder_map:
            remainder_map[remainder] = len(ans)
            remainder = remainder * 10
            ans.append(str(remainder // denominator))
            remainder = remainder % denominator
        if remainder:
            index = remainder_map[remainder]
            ans.insert(index, "(")
            ans.append(")")
        return "".join(ans)


if __name__ == '__main__':
    solution = Solution()
    numerator = 4
    denominator = 333
    print(solution.fractionToDecimal(numerator, denominator))
