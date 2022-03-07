class Solution:
    def letterCombinations(self, digits: str) -> list[str]:
        if digits == "":return []
        letter_map = {"2":"abc","3":"def","4":"ghi","5":"jkl","6":"mno","7":"pqrs","8":"tuv","9":"wxyz"}
        ans = [""]
        for i in digits:
            ans = [string+letter for string in ans for letter in letter_map[i]]
        return ans

if __name__ == '__main__':
    solution = Solution()
    digits = "23"
    print(solution.letterCombinations(digits))