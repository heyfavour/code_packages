import copy
class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        n_list = [str(i) for i in range(1,n+1)]
        res = []
        def helper(nums,n_list):
            if len(nums) == n:
                res.append(nums)
                if len(res) == k:return "".join(nums)
            for i in n_list:
                nn_list = copy.deepcopy(n_list)
                nn_list.remove(i)
                nums.append(i)
                result = helper(nums,nn_list)
                if result:return result
                nums.pop()
        return helper([],n_list)



if __name__ == '__main__':
    s = Solution()
    print(s.getPermutation(9,331987))
