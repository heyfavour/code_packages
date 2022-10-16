import functools


class Solution:
    def maxCoins(self, nums: list[int]) -> int:
        nums = [1] + nums + [1]
        @functools.cache
        def help(l, r):
            if l >= r - 1: return 0
            max_score = 0
            for i in range(l + 1, r):
                max_score = max(nums[l] * nums[i] * nums[r] + help(l, i) + help(i, r), max_score)
            return max_score

        return help(0, len(nums) - 1)


if __name__ == '__main__':
    nums = [3, 1, 5, 8]
    solution = Solution()
    print(solution.maxCoins(nums))
