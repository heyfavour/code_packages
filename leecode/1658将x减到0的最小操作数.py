from typing import List


class Solution:
    @classmethod
    def minOperations(self, nums: List[int], x: int) -> int:
        result = []
        nums.sort()
        max_len = len(nums)
        if sum(nums)>x:return -1
        def helper(nums, L, sum):
            print(L)
            if sum == x:
                result.append(L[:])
                return
            if sum > x:
                return
            if len(L) >= max_len:
                return
            for i in range(len(nums)):
                if i >= 1 and nums[i] == nums[i - 1]: continue
                num = nums[i]
                L.append(num)
                nums.pop(i)
                helper(nums, L,sum+num)
                nums.insert(i, num)
                L.pop()

        helper(nums, [], 0)
        return min([len(i) for i in result])


if __name__ == '__main__':
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    x = 134365
    print(Solution.minOperations(nums, x))
