class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        def partition(left,right):
            pivot = left
            for i in range(left+1,right+1):
                if nums[i]<nums[left]:
                    pivot = pivot+1
                    nums[i],nums[pivot] = nums[pivot],nums[i]
            nums[pivot],nums[left] = nums[left],nums[pivot]
            return pivot

        def quick_sort(left,right):
            if left<right:
                mid = partition(left,right)
                quick_sort(left,mid-1)
                quick_sort(mid+1,right)
        quick_sort(0,len(nums)-1)

        return nums


if __name__ == '__main__':
    nums = [2,3,4,1,3,5,7]
    solution = Solution()
    print(solution.findKthLargest(nums,1))
