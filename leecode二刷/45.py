class Solution:
    def jump(self, nums: list[int]) -> int:
        last_end,now_end = 0,0
        count = 0
        n = len(nums)
        for i in range(n):
            now_end = max((now_end,i+nums[i]))
            if now_end >= n-1:return count + 1
            if i == last_end:
                last_end = now_end
                count = count + 1

if __name__ == '__main__':
    solution = Solution()
    nums = [2, 3, 1, 1, 4]
    print(solution.jump(nums))