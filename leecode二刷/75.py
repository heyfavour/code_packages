class Solution:
    # # 快排
    # def sortColors(self, nums: list[int]) -> None:
    #     def partition(nums, start,end):
    #         pivot = start
    #         for i in range(start+1,end+1):
    #             if nums[i]<=nums[start]:
    #                 pivot = pivot+1
    #                 nums[i],nums[pivot] = nums[pivot],nums[i]
    #         nums[pivot],nums[start] = nums[start],nums[pivot]
    #         return pivot
    #     def quick_sort(nums, start, end):
    #         if start < end:
    #             mid = partition(nums,start,end)
    #             quick_sort(nums, start, mid - 1)
    #             quick_sort(nums, mid + 1, end)
    #
    #     quick_sort(nums, 0, len(nums) - 1)
    #     return nums

    # 桶排序
    def sortColors(self, nums: list[int]) -> None:
        bucket = [0]*3
        for i in nums:bucket[i] = bucket[i]+1
        i = 0
        for k,v in enumerate(bucket):
            nums[i:i+v] = [k]*v
            i = i+v
        return nums



if __name__ == '__main__':
    nums = [2, 0, 2, 1, 1, 0]
    solution = Solution()
    print(solution.sortColors(nums))
