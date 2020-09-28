from typing import  List
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_pos,end,step = 0,0,0
        for i in range(len(nums)):
            max_pos = max((i+nums[i],max_pos))
            print(max_pos)
            if i == max_pos and nums[i] == 0:
                if max_pos<len(nums):return False
        return True
if __name__ == '__main__':
    s = Solution()
    L = [2,0,0]
    print(s.canJump(L))
