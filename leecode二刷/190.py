class Solution:
    def reverseBits(self, n: int) -> int:
        ans = 0
        for i in range(32):
            ans = ans|(n&1)<<(31-i)#最有一位 左移
            n = n>>1
        return ans
