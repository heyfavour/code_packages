class solution:
    @classmethod
    def numdecodings(self, s: str) -> int:
        dp = [0] * len(s)
        for i in range(len(s)):
            if i == 0:
                dp[i] = 0 if s[0] == '0' else 1
            elif s[i] == '0':
                if int(s[i - 1:i + 1]) not in (10, 20):
                    return 0
                else:
                    dp[i] = dp[i - 2] if i - 2 >= 0 else 1
            else:
                if 10 < int(s[i - 1:i + 1]) <= 26:
                    dp[i] = dp[i - 1] + (dp[i - 2] if i - 2 >= 0 else 1)
                else:
                    dp[i] = dp[i - 1]
            print(dp)
        return dp[-1]


if __name__ == '__main__':
    print(solution.numdecodings('12'))
