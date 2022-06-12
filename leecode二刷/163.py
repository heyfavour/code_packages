class Solution:
    def findMissingRanges(self, nums: list[int], lower: int, upper: int) -> list[str]:
        ans = []
        p = lower
        nums = [lower-1]+nums+[upper+1]
        for i in nums:
            if i > p:
                if i == p+1:ans.append(str(p))
                else:ans.append(f"{p}->{i-1}")
            p = i+1
        return ans


if __name__ == '__main__':
    solution = Solution()
    nums = [0, 1, 3, 50, 75]
    lower = 0
    upper = 99
    print(solution.findMissingRanges(nums, lower, upper))
