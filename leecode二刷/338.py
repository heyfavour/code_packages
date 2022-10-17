class Solution:
    def countBits(self, n: int) -> list[int]:
        # return [bin(i).count("1") for i in range(n+1)]
        def count(n):
            c = 0
            while n>0:
                n = n&(n-1)
                c = c+1
            return c
        return [count(i) for i in range(n + 1)]