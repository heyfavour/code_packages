class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        start = abs(divisor) << 31
        cur = abs(dividend)
        res = 0

        for i in range(32):
            # 其实就是一直拿被除数减去除数，优化一下就是每次翻倍的减
            y = cur - (start >> i)
            if y >= 0:
                cur = y
                res += (1 << (31 - i))

                if res >= 2 << 30:
                    if (dividend > 0) != (divisor > 0):
                        return -(2 << 30)
                    else:
                        return (2 << 30) - 1

        if (dividend > 0) != (divisor > 0):
            res = -res

        return res
