class Solution:
    def numIslands(self, grid: list[list[str]]) -> int:
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1": grid[i][j] = "-1"
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def infect(x, y):
            nonlocal count
            if not (0 <= x < m) or not (0 <= y < n): return
            if grid[x][y] == "-1":
                grid[x][y] = count
            else:
                return
            for dx, dy in dirs:
                nx = x + dx
                ny = y + dy
                infect(nx,ny)

        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "-1":
                    count = count + 1
                    infect(i, j)
        return count


if __name__ == '__main__':
    grid = [
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],
        ["0", "0", "0", "1", "1"]
    ]
    solution = Solution()
    print(solution.numIslands(grid))
