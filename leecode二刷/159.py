from collections import defaultdict, deque


class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        n = len(s)
        max_len = 0
        char_dict = {}
        char_queue = deque()
        left = 0
        for i in range(n):
            print(char_queue,char_dict,left)
            char = s[i]
            char_dict[char] = i
            if char not in char_queue:  # 没有这个字母
                char_queue.append(char)
                if len(char_queue) >= 3:
                    fist_char = char_queue.popleft()
                    left = char_dict.pop(fist_char) + 1
            else:
                if char_queue[0] == char:
                    char_queue.append(char_queue.popleft())
            max_len = max(max_len, i - left + 1)
        return max_len


if __name__ == '__main__':
    solution = Solution()
    s = "abaccc"
    print(solution.lengthOfLongestSubstringTwoDistinct(s))
