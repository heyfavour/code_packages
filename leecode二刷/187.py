from collections import defaultdict


class Solution:
    def findRepeatedDnaSequences(self, s: str) -> list[str]:
        n = len(s)
        _dict = defaultdict(int)
        ans = []
        for i in range(9, n):
            string = s[i - 9:i + 1]
            _dict[string] = _dict[string] + 1
            if _dict[string] >= 2: ans.append(string)
        return ans


if __name__ == '__main__':
    solution = Solution()
    s = "AAAAAAAAAAA"
    print(solution.findRepeatedDnaSequences(s))
