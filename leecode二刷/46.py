class Solution:
    def permute(self, nums: list[int]) -> list[list[int]]:
        ans = ""
        n = len(nums)
        def backtrack(L,i):
            if L
            for i in enumerate(L):


        backtrack([],i)
if __name__ == '__main__':
    nums = [1, 2, 3]
    solution = Solution()
    print(solution.permute(nums))