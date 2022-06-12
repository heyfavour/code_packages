from functools import cmp_to_key


class Solution:
    def largestNumber(self, nums: list[int]) -> str:
        nums = list(map(str, nums))
        nums.sort(key=cmp_to_key(lambda x, y: int(x + y) - int(y + x)))
        return "".join(nums[::-1])


if __name__ == '__main__':
    solution = Solution()
    nums = [3, 30, 34, 5, 9]
    print(solution.largestNumber(nums))
