class Solution:
    def titleToNumber(self, columnTitle: str) -> int:
        char_dict = {chr(i): i - 64 for i in range(65, 91)}
        ans = 0
        while columnTitle:
            cur = columnTitle[0]
            ans = ans*26+char_dict[cur]
            columnTitle = columnTitle[1:]
        return ans


if __name__ == '__main__':
    solution = Solution()
    columnTitle = "AB"
    print(solution.titleToNumber(columnTitle))
