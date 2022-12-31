#26 删除有序数组中的重复项
#80 删除有序数组中的重复项
class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        i = 2
        for j in range(2,len(nums)):
            if nums[j]!=nums[i-2]:
                nums[i] = nums[j]
                i = i + 1
        return i


if __name__ == '__main__':
    nums = [0, 0, 1, 1, 1, 1, 2, 3, 3]
    solution = Solution()
    print(solution.removeDuplicates(nums))
