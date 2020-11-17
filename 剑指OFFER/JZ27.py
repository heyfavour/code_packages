# -*- coding:utf-8 -*-
class Solution:
    @classmethod
    def Permutation(self, ss):
        # write code here
        if ss == "": return []
        ss = list(ss)
        ss.sort()
        print(ss)
        len_ss = len(ss)
        result = []

        def helper(string, L):
            if L and len(L) == len_ss:
                result.append(L[:])
                return
            for i in range(len(string)):
                str_i = string[i]
                L.append(str_i)
                string.pop(i)
                helper(string, L)
                string.insert(i, str_i)
                L.pop()

        helper(ss, [])
        return result
if __name__ == '__main__':
    print(Solution.Permutation('abca'))
