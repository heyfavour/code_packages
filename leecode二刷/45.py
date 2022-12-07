class Solution:
    def jump(self, nums: list[int]) -> int:
        max_end,end,steps = 0,0,0
        n = len(nums)
        for i in range(n-1):
            if max_end>=i:
                max_end=max(max_end,i+nums[i])
                if i == end:
                    end = max_end
                    steps +=1
        return steps

if __name__ == '__main__':
    solution = Solution()
    nums = [2, 3, 1, 1, 4]
    print(solution.jump(nums))