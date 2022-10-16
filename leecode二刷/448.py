class Solution:
    def findDisappearedNumbers(self, nums: list[int]) -> list[int]:
        d = dict()
        for i in nums:
            d[i] = d.get(i,0)+1
        ans = []
        for i in range(1,len(nums)+1):
            if not d.get(i):ans.append(i)
        return ans
    """
        n = len(nums)
        for num in nums:
            x = (num - 1) % n
            nums[x] += n
        
        ret = [i + 1 for i, num in enumerate(nums) if num <= n]
        return ret
    """