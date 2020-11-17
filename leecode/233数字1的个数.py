class Solution:
    @classmethod
    def countDigitOne(self, n: int,x:int) -> int:
        str_n = str(n)
        len_str = len(str_n)
        dp = [0]*len_str
        for i in range(len_str):
            if i == 0:
                dp[0] = 1
            else:
                dp[i] = 10 ** (i) + 10 * dp[i - 1]
        count = 0
        for i in range(len_str-1,-1,-1):
            print(str_n[i])
            if i > x:count = count + dp[i-1]*(int(str_n[i])-1)+10**(i)
            if i == x:count = count + dp[i-1]*(int(str_n[i])-1) + 
            # if i < x:count = count + dp[i-1] + 1
        return count




if __name__ == '__main__':
    print(Solution.countDigitOne(899,1))
