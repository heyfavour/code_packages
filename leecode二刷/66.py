class Solution:
    def plusOne(self, digits: list[int]) -> list[int]:
        n = len(digits)
        add = 0
        for i in range(n - 1, -1, -1):
            _sum = digits[i] + (1 if i==n-1 else 0) + add
            add, num = _sum // 10, _sum % 10
            digits[i] = num
        if add: digits.insert(0, add)
        return digits


if __name__ == '__main__':
    solution = Solution()
    digits = [9, 9, 9]
    print(solution.plusOne(digits))
