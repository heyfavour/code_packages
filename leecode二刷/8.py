class Solution:
    def myAtoi(self, s: str) -> int:
        s = s.lstrip()
        ans = 0
        sign = 1
        sign_flag = False
        for i in s:
            if i  == "+" and not sign_flag:
                sign = 1
                sign_flag = True
            elif i == "-" and not sign_flag:
                sign = -1
                sign_flag = True
            elif i.isdigit():
                sign_flag = True
                ans = 10*ans+int(i)
            else:
                break
        ans =  sign*ans
        if ans>2**31 - 1:return 2**31 -1
        if ans<-2**31:return -2**31
        return ans

