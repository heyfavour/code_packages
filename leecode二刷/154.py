class Solution:
    def findMin(self, nums: list[int]) -> int:
        l, r = 0, len(nums) - 1
        while l < r:
            mid = (l + r) >> 1
            if nums[l] == nums[mid] == nums[r]:
                l = l + 1
                r = r - 1
            elif nums[mid] > nums[r]:  # 左侧有序
                l = mid + 1
            else:  # 右侧有序
                r = mid
        return nums[l]



if __name__ == '__main__':
    solution = Solution()
    nums = [2, 2, 2, 0, 1]
    print(solution.findMin(nums))
