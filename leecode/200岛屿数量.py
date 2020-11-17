from typing import List


class Solution:
    @classmethod
    def numIslands(self, grid: List[List[str]]) -> int:
        count = 0
        directions = [(1, 0), (-1, 0), (0, -1), (0, 1)]

        def helper(i, j):
            for (x, y) in directions:
                if 0 <= i + x < len(grid) and 0 <= j + y < len(grid[0]):
                    if grid[i + x][j + y] == '1':
                        grid[i + x][j + y] = '-1'
                        helper(i + x, j + y)
                    else:
                        continue

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    helper(i, j)
                    grid[i][j] = '-1'
                    count = count + 1
        return count


if __name__ == '__main__':
    grid = [
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],
        ["0", "0", "0", "1", "1"]
    ]
    print(Solution.numIslands(grid))
