class Solution:
    def findDuplicates(self, nums: list[int]) -> list[int]:
        for i in range(len(nums)):
            while nums[i] != nums[nums[i]-1]:
                nums[nums[i]-1],nums[i] = nums[i],nums[nums[i]-1]
        return [nums[i] for i in range(len(nums)) if nums[i]!=i+1]


if __name__ == '__main__':
    nums = [4, 3, 2, 7, 8, 2, 3, 1]
    solution=Solution()
    print(solution.findDuplicates(nums))
