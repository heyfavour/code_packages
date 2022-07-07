class Solution:
    def isStrobogrammatic(self, num: str) -> bool:
        reversed_map = {"6":"9","9":"6","8":"8","0":"0","1":"1"}
        l,r = 0,len(num)-1
        while l<=r:
            if num[l] not in reversed_map or reversed_map[num[l]]!=num[r]:return False
            l = l + 1
            r = r - 1
        return True