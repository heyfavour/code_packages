class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        if s == "":return 0
        count_str = 0
        for i in range(len(s)-1,-1,-1):
            if s[i] == " ":
                if count_str == 0:
                    continue
                break
            count_str = count_str+1

        return count_str
if __name__ == '__main__':
    s=Solution()
    print(s.lengthOfLastWord("a 1111"))
