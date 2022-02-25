class Solution:
    def intToRoman(self, num: int) -> str:
        roman_set = (
        ("M", 1000), ("CM", 900), ("D", 500), ("CD", 400),("C", 100),
        ("XC", 90), ("L", 50), ("XL", 40), ("X", 10),
        ("IX", 9), ("V", 5), ("IV", 4),("I", 1))
        roman_str = ""
        for k, v in roman_set:
            j = num // v
            num = num - j*v
            roman_str = roman_str + j * k
        return  roman_str


if __name__ == '__main__':
    num = 1994
    solution = Solution()
    print(solution.intToRoman(num))
