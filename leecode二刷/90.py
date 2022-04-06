class Solution:
    def subsetsWithDup(self, nums: list[int]) -> list[list[int]]:
        ans = []
        nums.sort()
        n = len(nums)
        def traceback(L,start):
            ans.append(L.copy())
            if len(L)== n:return
            for i in range(start,n):
                if i>start and nums[i] == nums[i-1]:continue
                L.append(nums[i])
                traceback(L,i+1)
                L.pop()
        traceback([],0)
        return ans

if __name__ == '__main__':
    solution =Solution()
    nums = [1, 2, 2]
    print(solution.subsetsWithDup(nums))

