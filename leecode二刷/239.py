import heapq
class Solution:
    def maxSlidingWindow(self, nums: list[int], k: int) -> list[int]:
        n = len(nums)
        ans = []
        max_heapq = []#最大堆维护
        for i in range(n):
            heapq.heappush(max_heapq,(-nums[i],i))#[num,i]
            while i-max_heapq[0][1] >=k:
                heapq.heappop(max_heapq)
            if i>=k-1:
                ans.append(-max_heapq[0][0])

        return ans
if __name__ == '__main__':
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    solution = Solution()
    print(solution.maxSlidingWindow(nums,k))
    #[3,3,5,5,6,7]