class Solution:
    def findPeakElement(self, nums: list[int]) -> int:
        nums = [float("-inf")] + nums + [float("-inf")]
        l, r = 1, len(nums) - 2
        while l < r:
            mid = (l + r) >> 1
            if nums[mid - 1] < nums[mid] > nums[mid + 1]:
                return mid - 1
            elif nums[mid - 1] < nums[mid]:
                l = l + 1
            else:
                r = r - 1
        return l - 1


if __name__ == '__main__':
    solution = Solution()
    nums = [1, 2, 1, 3, 5, 6, 4]
    print(solution.findPeakElement(nums))
