class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        left, right = 0, 0
        max_len = 1 if len(s) > 1 else 0
        while right < len(s) -1:
            if s[right + 1] in s[left:right]:
                max_len = max((max_len,right-left + 1))
                left +=1
            else:
                right +=1
        return max_len


if __name__ == '__main__':
    s = "abcabcbb"
    solution = Solution()
    max_len = solution.lengthOfLongestSubstring(s)
    print(max_len)
