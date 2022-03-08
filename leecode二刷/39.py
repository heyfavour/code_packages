class Solution:
    def combinationSum(self, candidates: list[int], target: int) -> list[list[int]]:
        ans = []
        def backtrack(L,idx):
            if sum(L) == target:ans.append(L.copy())
            if sum(L)>target:return False
            for i in range(idx,len(candidates)):
                L.append(candidates[i])
                backtrack(L,i)
                L.pop()
        backtrack([],0)
        return ans
if __name__ == '__main__':
    candidates = [2, 3, 6, 7]
    target = 7
    solution = Solution()
    print(solution.combinationSum(candidates,target))