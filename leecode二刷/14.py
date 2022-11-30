class Solution:
    def longestCommonPrefix(self, strs: list[str]) -> str:
        if not strs:return ""
        prefix = strs[0]
        for string in strs[1:]:
            prefix_len = min((len(prefix), len(string)))
            prefix = prefix[:prefix_len]
            for j in range(prefix_len):
                if prefix[j] != string[j]:
                    if j == 0:
                        return ""
                    else:
                        prefix = prefix[:j]
                        break
        return prefix



if __name__ == '__main__':
    solution = Solution()
    strs = ["flower", "flow", "flight"]
    print(solution.longestCommonPrefix(strs))

