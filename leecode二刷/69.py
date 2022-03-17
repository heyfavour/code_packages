class Solution:
    def mySqrt(self, x: int) -> int:
        if x <= 1: return x
        l, r = 0, x
        while l <= r:
            mid = (l + r) >> 1
            mid_sqrt = mid * mid
            if mid_sqrt == x:
                r = mid - 1
            elif mid_sqrt < x:
                l = mid + 1
            elif mid_sqrt > x:
                r = mid - 1
        return l-1


if __name__ == '__main__':
    x = 8
    solution = Solution()
    print(solution.mySqrt(x))
