class Solution:
    def restoreIpAddresses(self, s: str) -> list[str]:
        ans = []

        def traceback(s,L):
            if len(L)>4:return
            if len(L) == 4 and s == "":ans.append(".".join(L))
            for i in range(1,min((len(s)+1,4))):
                _ip = s[:i]
                if not _ip.isdigit() or (_ip[0] == "0" and _ip!="0"):break
                L.append(_ip)
                traceback(s[i:],L)
                L.pop()
        traceback(s,[])
        return ans


if __name__ == '__main__':
    solution = Solution()
    s = "101023"
    print(solution.restoreIpAddresses(s))
