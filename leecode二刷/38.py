class Solution:
    def countAndSay(self, n: int) -> str:
        def desc_num(num):
            string_num = str(num)
            sign_num = "_"
            string = ""
            for i, v in enumerate(string_num):
                print(i,v)
                if i == 0:
                    count = 1
                    sign_num = v
                elif v != sign_num:
                    string = string + str(count) + str(sign_num)
                    count = 1
                    sign_num = v
                else:
                    count = count+1
            string = string + str(count) + str(sign_num)
            return int(string)

        dp = [0] * n
        for i in range(n):
            if i == 0:
                dp[0] = 1
            else:
                dp[i] = desc_num(dp[i - 1])
        return dp[-1]


if __name__ == '__main__':
    n = 4
    solution = Solution()
    print(solution.countAndSay(n))
