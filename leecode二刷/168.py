class Solution:
    def convertToTitle(self, columnNumber: int) -> str:
        """
        A-1-0  (X-1)%26
        ...
        Z-26-25
        """
        ans = ""
        while columnNumber:
            cur = (columnNumber-1) % 26
            ans = chr(cur+65)+ans
            columnNumber = (columnNumber - 1) // 26
        return ans


if __name__ == '__main__':
    solution = Solution()
    columnNumber = 701
    print(solution.convertToTitle(columnNumber))
