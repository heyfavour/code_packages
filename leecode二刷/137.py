class Solution:
    """
    x&~x = 0
    x&_0 = x
    """
    def singleNumber(self, nums: list[int]) -> int:
        once, twice = 0, 0
        for i in nums:
            once = (once ^ i) & ~twice
            twice = (twice ^ i) & ~once
        return once


if __name__ == '__main__':
    nums = [2, 2, 3, 2]
    solution = Solution()
    print(solution.singleNumber(nums))
