from typing import List

class Solution:
    #1.贪心算法  区间/时间 覆盖  最优解
    def jump(self, nums: List[int]) -> int:
        #maxpos 目前最远的边界
        #end    目前的边界
        maxpos = end = step =  0
        for i in range(len(nums)-1):
            maxpos = max(maxpos,i + nums[i])
            if i == end:
                end = maxpos
                step = step + 1
        return step



if __name__ == '__main__':
    l = [2,3,1,1,4]
    s = Solution()
    print(s.jump(l))
