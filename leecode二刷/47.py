class Solution:
    def permuteUnique(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        ans = []

        def backtrack(L, l):
            if L == []: ans.append(l.copy())
            for k, v in enumerate(L):
                if k >= 1 and L[k] == L[k - 1]: continue
                L.pop(k)
                l.append(v)
                backtrack(L, l)
                L.insert(k, v)
                l.pop()

        backtrack(nums, [])
        return ans


if __name__ == '__main__':
    solution = Solution()
    nums = [1, 1, 2]
    print(solution.permuteUnique(nums))
