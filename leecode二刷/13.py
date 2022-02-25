class Solution:
    def romanToInt(self, s: str) -> int:
        roman_dict = {"M":1000,"CM":900,"D":500,"CD":400,"C":100,"XC":90,"L":50,"XL":40,"X":10,"IX":9,"V":5,"IV":4,"I":1}
        roman_int = 0
        len_s = len(s)
        i = 0
        while i<len_s:
            if roman_dict.get(s[i:i+2]) is None:
                roman_int = roman_int + roman_dict[s[i]]
                i = i + 1
            else:
                roman_int = roman_int + roman_dict[s[i:i+2]]
                i = i + 2
        return roman_int

if __name__ == '__main__':
    s = "MCMXCIV"
    solution = Solution()
    print(solution.romanToInt(s))