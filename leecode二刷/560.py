from collections import defaultdict
class Solution:
    def subarraySum(self, nums: list[int], k: int) -> int:
        n = len(nums)
        prefix = [0]*n
        prefix_dict = defaultdict(int)
        prefix_dict[0] = 1
        ans = 0
        for i in range(n):
            prefix[i] = prefix[i-1]+nums[i]
            ans = ans + prefix_dict[prefix[i]-k]
            prefix_dict[prefix[i]] = prefix_dict[prefix[i]]+1
        return ans

if __name__ == '__main__':
    nums = [1, 1, 1]
    k = 2
    solution = Solution()
    print(solution.subarraySum(nums,k))