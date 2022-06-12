class Solution:
    def maxPoints(self, points: list[list[int]]) -> int:
        _dict = dict()  # (k,b)
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if points[j][0] - points[i][0] == 0:  # x=n
                    L = _dict.get(points[j][0], [])
                    if points[i] not in L: _dict.setdefault(points[j][0], []).append(points[i])
                    if points[j] not in L: _dict.setdefault(points[j][0], []).append(points[j])

                else:
                    k = (points[j][1] - points[i][1]) * 1.0 / (points[j][0] - points[i][0])
                    b = points[j][1] - k * points[j][0]
                    L = _dict.get((k, b), [])
                    if points[i] not in L: _dict.setdefault((k, b), []).append(points[i])
                    if points[j] not in L: _dict.setdefault((k, b), []).append(points[j])

        return max([len(i) for i in _dict.values()])


if __name__ == '__main__':
    solution = Solution()
    points = [[1, 1], [3, 2], [5, 3], [4, 1], [2, 3], [1, 4]]
    print(solution.maxPoints(points))
