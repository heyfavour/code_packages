from typing import List


class Solution:
    @classmethod
    def partition(cls, nums, start, end):
        privot = start
        for i in range(start + 1, end + 1):
            if nums[i] <= nums[start]:
                privot = privot + 1
                nums[privot], nums[i] = nums[i], nums[privot]
        nums[privot], nums[start] = nums[start], nums[privot]
        return privot

    @classmethod
    def quick(cls, nums, start, end):
        if start < end:
            mid = cls.partition(nums, start, end)
            cls.quick(nums, start, mid - 1)
            cls.quick(nums, mid + 1, end)
        return nums

    @classmethod
    def countSort(self, nums):
        countnums = [0 for i in range(101)]
        for i in nums:
            countnums[i] = countnums[i] + 1
        new_nums = []
        for i in range(101):
            if countnums[i] == 0:
                pass
            else:
                new_nums = new_nums + [i,] * countnums[i]
        return new_nums

    @classmethod
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        # sort_nums = self.quick(nums, 0, (len(nums) - 1))
        sort_nums = self.countSort(nums)
        sort_dict = {}
        j = 0
        for i in range(len(sort_nums)):
            if sort_dict.get(sort_nums[i]) is None:
                sort_dict[sort_nums[i]] = j
            j = j + 1
        return [sort_dict[i] for i in nums]


if __name__ == '__main__':
    nums = [8,1,2,2,3]
    print(Solution.smallerNumbersThanCurrent(nums))
