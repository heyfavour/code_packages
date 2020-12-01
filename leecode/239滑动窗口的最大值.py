from typing import List

from collections import deque


class Solution:
    @classmethod
    def maxSlidingWindow(self, nums: 'List[int]', k: 'int') -> 'List[int]':
        # base cases
        d = deque()
        result = []
        for i in range(len(nums)):
            while d and nums[d[-1]] <= nums[i]:
                d.pop()
            d.append(i)
            if d[0] < i + 1 - k: d.popleft()
            if i+1 >= k: result.append(nums[d[0]])
        return result


if __name__ == '__main__':
    nums = [2,3,4,2,6,2,5,1]
    k = 3
    print(Solution.maxSlidingWindow(nums, k))
