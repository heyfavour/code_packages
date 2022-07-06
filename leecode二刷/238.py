class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        L,R,ans = [1]*n,[1]*n,[1]*n

        for i in range(n):
            if i == 0:L[i] = nums[0]
            else:L[i] = L[i-1]*nums[i]
        for i in range(n-1,-1,-1):
            if i == n-1:R[i] = nums[i]
            else:R[i] = R[i+1]*nums[i]
        for i in range(n):
            if i == 0:ans[i] = R[i+1]
            elif i == n-1:ans[i] = L[i-1]
            else:ans[i] = L[i-1]*R[i+1]
        return ans