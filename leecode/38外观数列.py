class Solution:
    def countAndSay(self, n: int) -> str:
        if n == 1:return "1"
        if n >= 1:
            return self.describe_str(self.countAndSay(n-1))

    def describe_str(self,intstr):
        now_num=intstr[0]
        describe_str = ""
        count_num = 0
        for i in intstr:
            if now_num ==  i:
                count_num = count_num + 1
            else:
                describe_str = describe_str + str(count_num) + str(now_num)
                now_num = i
                count_num = 1
        describe_str = describe_str + str(count_num) + str(now_num)
        return describe_str





if __name__ == '__main__':
    s = Solution()
    print(s.countAndSay(5))

