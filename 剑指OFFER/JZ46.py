# -*- coding:utf-8 -*-
class Solution:
    @classmethod
    def LastRemaining_Solution(self, n, m):
        # write code here
        nlist = [i for i in range(n)]
        p = 0
        while len(nlist) > 1:
            p = ((m + p) % len(nlist)) - 1
            if p < 0: p = p + len(nlist)
            nlist.pop(p)
        return nlist[0]


if __name__ == '__main__':
    print(Solution.LastRemaining_Solution(5, 3))
