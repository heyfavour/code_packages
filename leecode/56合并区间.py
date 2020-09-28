from typing import List


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
        result = []
        for i in intervals:
            if result == [] or i[0] > result[-1][1]:
                result.append(i)
            else:
                result[-1][1] = max(i[1], result[-1][1])

        return result


if __name__ == '__main__':
    s = Solution()
    intervals = [[1, 3], [4, 6], [8, 10], [15, 18], [1, 2]]
    print(s.merge(intervals))
