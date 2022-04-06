import collections


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        _dict, n = {}, len(t)
        for i in t: _dict[i] = _dict.get(i, 0) + 1
        i,res = 0,(0,float("inf"))
        for k,v in enumerate(s):
            if _dict.get(v,0)>0:n=n-1#如果缺这个数 n-1
            _dict[v] = _dict.get(v,0) - 1#这个数个数-1
            if n == 0:#走到底部
                while True:
                    c = s[i]
                    if _dict[c] == 0:break#遇到t中的数字且不能再缩进了
                    _dict[c] = _dict[c]+1#个数加一
                    i = i+1#向前缩进
                if k-i<(res[1]-res[0]):res = (i,k)
                i=i+1
                n=n+1
                _dict[c] = _dict[c]+1
        return ''  if res[1]>len(s) else s[res[0]:res[1]+1]



if __name__ == '__main__':
    s = "ADOBECODEBANC"
    t = "ABC"
    solution = Solution()
    print(solution.minWindow(s, t))
