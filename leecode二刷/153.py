class Solution:
    def findMin(self, nums: list[int]) -> int:
        l,r = 0,len(nums)-1
        while l<r:
            mid = (l+r)>>1
            if nums[mid]>nums[r]:#高处
                l = mid+1
            else:
                r = mid
        return nums[l]


if __name__ == '__main__':
    solution = Solution()
    nums = [3, 1, 2]
    print(solution.findMin(nums))
