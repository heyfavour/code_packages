# -*- coding:utf-8 -*-
from functools import cmp_to_key


class Solution:
    @classmethod
    def PrintMinNumber(self, numbers):
        # write code here
        key = cmp_to_key(lambda x, y: 1 if str(x) + str(y) > str(y) + str(x) else -1)
        L = sorted(numbers, key=key)
        return "".join([str(i) for i in L])


if __name__ == '__main__':
    L = [3, 32, 321]
    print(Solution.PrintMinNumber(L))
