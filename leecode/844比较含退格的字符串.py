class Solution:
    @classmethod
    def backspaceCompare(self, S: str, T: str) -> bool:
        def helper(string):
            L = []
            for i in string:
                if i == "#":L.pop()
                else:L.append(i)
            return  str(L)
        return helper(S) == helper(T)

if __name__ == '__main__':
    S = "ab#c"
    T = "ad#c"
    print(Solution().backspaceCompare(S,T))

