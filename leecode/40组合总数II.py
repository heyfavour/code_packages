class Solution(object):
    def combinationSum(self, candidates, target):
        candidates.sort()
        result = []
        def back(position,path):
            if sum(path) == target:
                result.append(path)
                return
            for i in range(position,len(candidates)):
                print(position,i,path)
                if sum(path) + candidates[i] > target: break
                if i > position and candidates[i] == candidates[i-1]:
                    continue
                back(i + 1,path+[candidates[i]])

        back(0,[])
        return result

if __name__ == '__main__':
    s = Solution()
    l = [1,1,1,1]
    #l = [2,2,1,2]
    print(s.combinationSum(l, 8))

