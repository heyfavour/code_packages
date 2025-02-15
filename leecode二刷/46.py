class Solution:
    def permute(self, nums: list[int]) -> list[list[int]]:
        ans = []
        def backtrack(L, l):
            if L == []: ans.append(l.copy())
            for k, v in enumerate(L):
                L.pop(k)
                l.append(v)
                backtrack(L,l)
                L.insert(k, v)
                l.pop()
        backtrack(nums, [])
        return ans


if __name__ == '__main__':
    nums = [1, 2, 3]
    solution = Solution()
    print(solution.permute(nums))
