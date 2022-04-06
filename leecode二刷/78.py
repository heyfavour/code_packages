class Solution:
    def subsets(self, nums: list[int]) -> list[list[int]]:
        ans = []

        def backtrack(L, i):
            ans.append(L.copy())
            if len(L) >= len(nums): return
            for j in range(i, len(nums)):
                L.append(nums[j])
                backtrack(L, j + 1)
                L.pop()

        backtrack([], 0)
        return ans

if __name__ == '__main__':
    nums = [1, 2, 3]
    solutin = Solution()
    print(solutin.subsets(nums))
