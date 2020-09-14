from typing import List

class Solution:
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height) - 1
        ans = 0
        while left <right:
            area = min(height[left],height[right])*(right-left)
            ans = max(area,ans)
            if height[left] <= height[right]:
                left = left + 1
            else:
                right = right - 1
        return ans


if __name__ == '__main__':
    l = [1,8,6,2,5,4,8,3,7]
    s = Solution()
    print(s.maxArea(l))
