class Solution:
    @classmethod
    def isHappy(self, n: int) -> bool:
        result = []

        def helper(nums):
            sum_nums = 0
            for i in str(nums):
                sum_nums = sum_nums + int(i) ** 2
            if sum_nums == 1: return True
            if sum_nums in result:
                return False
            else:
                result.append(sum_nums)
                return helper(sum_nums)

        return helper(n)


if __name__ == '__main__':
    print(Solution.isHappy(19))
