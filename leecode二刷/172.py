class Solution:
    def trailingZeroes(self, n: int) -> int:
        # 2 4 5 6 8
        count = 0
        for i in range(n):
            while i % 5 == 0:
                i = i // 5
                count = count + 1
        return count


if __name__ == '__main__':
    solution = Solution()
    print(solution.trailingZeroes(30))
