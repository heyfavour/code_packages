class Solution:
    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        _dict = {}
        for i in strs:_dict.setdefault("".join(sorted(list(i))),[]).append(i)
        return list(_dict.values())
if __name__ == '__main__':
    solution = Solution()
    strs = ["eat","tea","tan","ate","nat","bat"]
    print(solution.groupAnagrams(strs))