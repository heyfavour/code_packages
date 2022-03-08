class Solution:
    def combinationSum2(self, candidates: list[int], target: int) -> list[list[int]]:
        ans=[]
        candidates.sort()
        n = len(candidates)
        def backtrack(L,idx):
            if sum(L) == target:ans.append(L.copy())
            if sum(L) > target:return False
            for i in range(idx,n):
                if i>idx and candidates[i] == candidates[i-1]:continue
                L.append(candidates[i])
                backtrack(L,i+1)
                L.pop()

        backtrack([],0)
        return ans


if __name__ == '__main__':
    solution = Solution()
    candidates = [10, 1, 2, 7, 6, 1, 5]
    target = 8
    print(solution.combinationSum2(candidates,target))