import math

class Solution:
    def countPrimes(self, n: int) -> int:#0 1 X 2是第一个质数
        if n<=2:return 0
        nums = [1]*n
        nums[0],nums[1] = 0,0
        for i in range(2,int(math.sqrt(n)+1)):
            if nums[i] == 1:nums[i * i:n:i] = [0] * len(nums[i * i:n:i])
        return sum(nums)


if __name__ == '__main__':
    n = 10
    solution = Solution()
    print(solution.countPrimes(n))
