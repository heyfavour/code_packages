class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t): return False

        def string_to_num(string):
            dict_s, nums_s, ns = {}, 0, ""
            for i in range(len(string)):
                if dict_s.get(string[i]) is None:
                    nums_s = nums_s + 1
                    dict_s[string[i]] = str(nums_s)
                ns = ns + dict_s[string[i]]
            return ns

        return string_to_num(s) == string_to_num(t)


if __name__ == '__main__':
    solution = Solution()
    s = "egg"
    t = "add"
    print(solution.isIsomorphic(s, t))
