class Solution:
    def wordBreak(self, s: str, wordDict: list[str]) -> list[str]:
        ans = []

        def dfs(s, path):
            if s == "": ans.append(path.copy())
            for i in wordDict:
                if i != s[:len(i)]: continue
                path.append(i)
                dfs(s[len(i):], path)
                path.pop()

        dfs(s, [])
        return [" ".join(path) for path in ans]


if __name__ == '__main__':
    solution = Solution()
    s = "catsanddog"
    wordDict = ["cat", "cats", "and", "sand", "dog"]
    print(solution.wordBreak(s, wordDict))
