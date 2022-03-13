class Solution:
    def merge(self, intervals: list[list[int]]) -> list[list[int]]:
        intervals.sort()
        result = []
        for i in range(len(intervals)):
            if result == [] or intervals[i][0] > result[-1][1]:
                result.append(intervals[i])
            else:
                result[-1][1] = max(result[-1][1],intervals[i][1])
        return result

if __name__ == '__main__':
    solution = Solution()
    intervals = [[1, 3], [2, 6], [8, 10], [15, 18], [9, 11]]
    print(solution.merge(intervals))
