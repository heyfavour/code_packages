from typing import List


class Solution:
    @classmethod
    def majorityElement(self, nums: List[int]) -> int:
        d = {}
        for i in range(len(nums)):
            d[nums[i]] = d.get(nums[i], 0) + 1
        return max(d,key=d.get)


if __name__ == '__main__':
    nums = [3,3,4]
    print(Solution.majorityElement(nums))
