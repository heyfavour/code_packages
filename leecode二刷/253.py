import heapq
class Solution:
    def minMeetingRooms(self, intervals: list[list[int]]) -> int:
        ans = 0
        intervals = sorted(intervals,key=lambda x:x[0])
        n = len(intervals)
        stack = []##END  (START,END)
        for i in range(n):
            while stack and stack[0][1][1] <= intervals[i][0]:#能结束的全部结束
                heapq.heappop(stack)
            heapq.heappush(stack,(intervals[i][1],intervals[i]))#最早结束的排前面
            ans = max(ans,len(stack))
        return ans