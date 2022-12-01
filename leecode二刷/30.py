import copy
from collections import Counter


class Solution:
    """
    def findSubstring(self, s: str, words: list[str]) -> list[int]:
        words_list = []
        def dfs_string(words,string):
            if words == []:words_list.append(string)
            for k,v in enumerate(words):
                nstring = string + v
                nwords = copy.deepcopy(words)
                nwords.pop(k)
                dfs_string(nwords,nstring)

        dfs_string(words,"")
        words_len = len(words_list[0])
        ans = []
        for i in range(len(s)):
            if str(s[i:i+words_len]) in words_list:
                ans.append(i)
        return ans
        """
    """
    def findSubstring(self, s: str, words: list[str]) -> list[int]:
        words_counter = Counter(words)
        word_len = len(words[0])
        words_num = len(words)
        ans = []
        for i in range(len(s)):
            string = [s[j:j + word_len] for j in range(i, i + words_num * word_len, word_len)]
            if Counter(string) == words_counter: ans.append(i)
        return ans
    """



if __name__ == '__main__':
    s = "barfoothefoobarman"
    words = ["foo", "bar"]
    solution = Solution()
    print(solution.findSubstring(s, words))
