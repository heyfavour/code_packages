class Solution:
    def searchInsert(self, nums: list[int], target: int) -> int:
        if not nums: return 0
        l, r = 0, len(nums)-1
        while l <= r:
            mid = (l + r) >> 1
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = l + 1
            elif nums[mid] > target:
                r = r - 1
        return l


if __name__ == '__main__':
    solution = Solution()
    nums = [1, 3, 5, 6]
    target = 7
    print(solution.searchInsert(nums,target))