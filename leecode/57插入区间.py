from typing import List

class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        intervals.append(newInterval)
        intervals.sort(key=lambda x:x[0])
        n  = []
        for i in intervals:
            if len(n) == 0 or i[0]>n[-1][1]:
                n.append(i)
            else:
                n[-1][1] = max(i[-1],n[-1][1])
        return n

if __name__ == '__main__':
    intervals = [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]]
    newInterval = [4, 8]
    s = Solution()
    print(s.insert(intervals,newInterval))
