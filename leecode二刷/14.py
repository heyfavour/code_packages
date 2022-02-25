class Solution:
    def longestCommonPrefix(self, strs: list[str]) -> str:
        len_strs = len(strs)
        if len_strs == 0:return ""
        common_pre = strs[0]
        for str2 in strs[1:]:
            _min = min((len(common_pre), len(str2)))
            common_pre = common_pre[:_min]
            for j in range(1,_min+1):
                if common_pre[:j] != str2[:j]:
                    if j == 1:return ""
                    else:
                        common_pre = common_pre[:j-1]
                        continue
        return common_pre



if __name__ == '__main__':
    solution = Solution()
    strs = ["flower", "flow", "flight"]
    print(solution.longestCommonPrefix(strs))

