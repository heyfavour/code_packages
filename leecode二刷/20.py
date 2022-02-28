class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        _dict = {"(":")","{":"}","[":"]"}
        match = False
        for i in s:
            if i in _dict.keys():
                stack.append(i)
            elif stack == []:
                return False
            else:
                left = stack.pop()
                if _dict[left] != i:return False
        return stack == []
