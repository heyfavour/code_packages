from typing import List
class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        nums.sort()
        for i in range(len(nums)-k+1):
            print(i)
if __name__ == '__main__':
    solution = Solution()
    nums = [9, 4, 1, 7]
    k = 2
    print(solution.minimumDifference(nums,k))