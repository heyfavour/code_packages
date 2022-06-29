class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        ans = []
        def dfs(nums,i):
            if len(nums) == k and sum(nums) == n:
                ans.append(nums.copy())
                return
            if len(nums)>=k or  sum(nums)>=n:return
            for j in range(i,10):
                nums.append(j)
                dfs(nums,j+1)
                nums.pop()

        dfs([],1)
        return ans