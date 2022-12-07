import copy
from collections import Counter


class Solution:
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

    def findSubstring(self, s: str, words: list[str]) -> list[int]:
        ans = []
        cnt, n, l = len(words), len(words[0]), len(s)
        for i in range(n):
            if i + cnt * n > l: break
            diff = Counter()

            for word in words: diff[word] -= 1
            for p in range(i, l, n):  # 滑动窗口 每次划过1个字符
                word = s[p:p + n]
                if p < i + cnt * n:
                    diff[word] += 1
                else:
                    diff[word] += 1
                    word = s[p - cnt * n:p - (cnt-1) * n]
                    diff[word] -= 1
                if [i != 0 for i in diff.values()].count(True) == 0: ans.append(p - (cnt-1) * n)
        return ans


if __name__ == '__main__':
    s = "barfoothefoobarman"
    words = ["foo", "bar"]
    solution = Solution()
    print(solution.findSubstring(s, words))
