class Solution:
    def shortestPalindrome(self, s: str) -> str:
        """
        s 匹配 s[::-1] 回文部分则由kmp最后回退的位置决定
        |s|--- ---|S|
        """
        def build_next(patt):
            #ABABC
            next = [0]
            prefix_len = 0#当前最长公共长度
            i=1
            while i<len(patt):
                if patt[i] == patt[prefix_len]:
                    prefix_len = prefix_len+1
                    i = i+1
                    next.append(prefix_len)
                else:
                    if prefix_len == 0:
                        i=i+1
                        next.append(prefix_len)
                    else:
                        prefix_len = next[prefix_len-1]#位次=长度-1
            return next


        # print(build_next("ABABC"))
        patt = s
        string = s[::-1]
        next = build_next(patt)
        i,j = 0,0
        while i<len(string):
            if string[i] == patt[j]:
                i=i+1
                j=j+1
            elif j>0:
                j = next[j-1]
            else:
                i=i+1
            if j == len(patt):
                return patt
        n = len(string)
        return string[:n-j]+patt




if __name__ == '__main__':
    s = "baab"
    solution = Solution()
    print(solution.shortestPalindrome(s))