class Solution:
    """
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        ans = False

        def dfs(s1, s2, s3):
            nonlocal ans
            if s1 and s1[0] == s3[0]: dfs(s1[1:], s2, s3[1:])
            if s2 and s2[0] == s3[0]: dfs(s1, s2[1:], s3[1:])
            if s1 == s2 == s3 == "":
                nonlocal ans
                ans = True

        dfs(s1, s2, s3)
        return ans
    """

    # dp
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        m, n, o = len(s1), len(s2), len(s3)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        for i in range(m):
            if s1[i] != s3[i]: break
            dp[i + 1][0] = True
        for i in range(n):
            if s2[i] != s3[i]: break
            dp[0][i + 1] = True
        for i in range(m):
            for j in range(n):
                dp[i + 1][j + 1] = (dp[i][j + 1] and s1[i] == s3[i + j + 1]) or (dp[i + 1][j] and s2[j] == s3[i + j + 1])
        return dp[-1][-1]


if __name__ == '__main__':
    solution = Solution()
    s1 = "aabcc"
    s2 = "dbbca"
    s3 = "aadbbcbcac"

    s1 = "cbcccbabbccbbcccbbbcabbbabcababbbbbbaccaccbabbaacbaabbbc"
    s2 = "abcbbcaababccacbaaaccbabaabbaaabcbababbcccbbabbbcbbb"
    s3 = "abcbcccbacbbbbccbcbcacacbbbbacabbbabbcacbcaabcbaaacbcbbbabbbaacacbbaaaabccbcbaabbbaaabbcccbcbabababbbcbbbcbb"
    print(solution.isInterleave(s1, s2, s3))
