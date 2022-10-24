class Solution:
    def findUnsortedSubarray(self, nums: list[int]) -> int:
        n = len(nums)
        max = nums[-1]
        first = False
        l = n-1
        for i in range(n-1,-1,-1):
            if i>=1 and nums[i]<nums[i-1]:
                max = nums[i]
                first = True
            if nums[i]<max and first:
                l = i
                first = False
        min = nums[0]
        first = False
        r = 0
        for i in range(n):
            if i<=n-2 and nums[i]>nums[i+1]:
                min = nums[i]
                first = True
            if nums[i]>min and first:
                r = i
                first = False
        print(l,r)
        if l>=r:return 0
        return r-l-1

if __name__ == '__main__':
    solution = Solution()
    #print(solution.findUnsortedSubarray(nums = [2,6,4,8,10,9,15]))
    print(solution.findUnsortedSubarray(nums = [2,1]))