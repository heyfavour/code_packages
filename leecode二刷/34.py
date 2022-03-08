class Solution:
    def searchRange(self, nums: list[int], target: int) -> list[int]:
        n = len(nums)
        L,R = 0,n-1
        #左边界
        while L<=R:
            mid = (L+R)>>1
            if nums[mid] == target:
                R = mid -1
            elif nums[mid] > target:
                L = mid+1
            else:
                R = mid-1
        left = L
        #右边界
        while L<=R:
            mid = (L+R)>>1
            if nums[mid] == target:
                L = mid
            elif nums[mid]<target:
                L = mid
            elif nums[mid]>target:
                R = mid + 1
        right = R
        return [left,right]

if __name__ == '__main__':
    nums = [5, 7, 7, 8, 8, 10]
    target = 8
    print(Solution.searchRange(nums, target))
