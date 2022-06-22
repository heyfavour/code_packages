from collections import defaultdict


class Solution:
    def isHappy(self, n: int) -> bool:
        sum_dict = defaultdict(int)
        while n != 1:
            if sum_dict[n] >= 2: return False
            n = str(n)
            sum = 0
            for i in n:
                sum = sum + int(i) ** 2
            n = sum
            sum_dict[n] = sum_dict[n] + 1
        return True


if __name__ == '__main__':
    solution = Solution()
    n = 19
    print(solution.isHappy(n))
