class Solution:
    def isScramble(self, s1: str, s2: str) -> bool:
        import functools
        @functools.cache
        def dfs(s1,s2):
            if s1==s2:return True
            if sorted(s1) !=sorted(s2):return False
            for i in range(1,len(s1)):
                if dfs(s1[:i],s2[:i])  and dfs(s1[i:],s2[i:]):return True
                if dfs(s1[:i],s2[-i:]) and dfs(s1[i:],s2[:-i]):return True
            return False
        return dfs(s1,s2)


if __name__ == '__main__':
    s1 = "abb"
    s2 = "bba"
    solution = Solution()
    print(solution.isScramble(s1,s2))
