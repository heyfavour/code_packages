import functools


class Solution:
    def wordBreak(self, s: str, wordDict: list[str]) -> bool:
        wordDict.sort(key=lambda x: -len(x))
        @functools.cache
        def dfs_help(s):
            if s == "":return True
            for i in wordDict:
                if i != s[:len(i)]:continue
                if dfs_help(s[len(i):]):return True
            return False
        return dfs_help(s)

if __name__ == '__main__':
    wordDict = ["a","aa","aaa","aaaa","aaaaa","aaaaaa","aaaaaaa","aaaaaaaa","aaaaaaaaa","aaaaaaaaaa"]
    s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"
    #wordDict = ["leet", "code"]
    solution = Solution()
    print(solution.wordBreak(s,wordDict))