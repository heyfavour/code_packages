class Solution:
    @classmethod
    def countDigitOne(self, n: int, x: int) -> int:
        count = 0
        num = n
        base = 1
        while base < n:
            pre, k, after = num // 10, num % 10, n % base
            if k > x: count = count + (pre + 1) * base
            if k < x: count = count + pre * base
            if k == x: count = count + pre * base + after + 1
            num = num // 10
            base = base * 10
        return count


if __name__ == '__main__':
    print(Solution.countDigitOne(2593, 1))
