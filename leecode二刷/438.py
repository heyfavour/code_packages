from collections import defaultdict


class Solution:
    def findAnagrams(self, s: str, p: str) -> list[int]:
        p_dict = defaultdict(int)
        for char in p: p_dict[char] = p_dict[char] + 1
        m, n = len(p), len(s)
        l, ans, tp_dict = 0, [], defaultdict(int)
        for r in range(n):
            char = s[r]
            if p_dict.get(char) == None:  # 字母不存在
                tp_dict = defaultdict(int)
                l = r + 1
            else:  # 字母存在
                tp_dict[char] = tp_dict[char] + 1
                if tp_dict[char] > p_dict[char]:
                    while True:
                        tp_dict[s[l]] = tp_dict[s[l]] - 1
                        l = l + 1
                        if s[l - 1] == char: break
                else:
                    if (r - l + 1) < m:
                        pass
                    elif (r - l + 1) == m:
                        ans.append(l)
                        tp_dict[s[l]] = tp_dict[s[l]] - 1
                        l = l + 1
                    elif (r - l) > m:
                        tp_dict[s[l]] = tp_dict[s[l]] - 1
                        l = l + 1

        return ans


if __name__ == '__main__':
    solution = Solution()
    s = "abacbabc"
    p = "abc"
    print(solution.findAnagrams(s, p))
