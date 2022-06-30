class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        from collections import defaultdict
        count = defaultdict(int)
        for i in nums:
            count[i] = count[i]+1
            if count[i]>=2:return True
        return False