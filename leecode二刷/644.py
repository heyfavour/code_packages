class Solution:
    def findMaxAverage(self, nums: list[int], k: int) -> float:
        avg_min, avg_max = min(nums), max(nums)
        n = len(nums)

        def check(avg):
            # 判断存在去间 sum(每个数-avg)>0 说明有比这个数更大的数 否则应该为0
            r_presum, min_presum = 0, 0
            for i in range(k): r_presum += nums[i] - avg
            if r_presum >= 0: return True
            l_presum = 0
            for i in range(k, n):
                r_presum += nums[i] - avg
                l_presum += nums[i - k] - avg
                min_presum = min(l_presum, min_presum)
                if r_presum - min_presum >= 0: return True
            return False

        while avg_max - avg_min > 10 ** (-5):
            mid = (avg_max + avg_min) / 2
            if check(mid):
                avg_min = mid
            else:
                avg_max = mid
        print(avg_min)
        return avg_min


if __name__ == '__main__':
    solution = Solution()
    solution.findMaxAverage([1, 2, 3, 4, 5], 2)
