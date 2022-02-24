class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)

        def is_match(i, j):
            if j == n: return i == m  # p到了最后，i如果也到最后则匹配否则不匹配
            first_char_match = (i < m) and (p[j] == '.' or p[j] == s[i])  # i,j对位匹配
            if j + 1 <= n-1 and p[j + 1] == "*":# p下一位如果是*
                return is_match(i, j + 2) or is_match(i + 1, j)#0 为 N位 s[i] == p[j+2] s[i+1]=p[j]
            return first_char_match and is_match(i + 1, j + 1)

        return is_match(0, 0)

if __name__ == '__main__':
    s = "ab"
    p = ".*"
    solution = Solution()
    print(solution.isMatch(s,p))
