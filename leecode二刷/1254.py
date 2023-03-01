class Solution:
    def closedIsland(self, grid: list[list[int]]) -> int:
        dirs = [(0,1),(0,-1),(1,0),(-1,0)]
        m ,n = len(grid),len(grid[0])
        def dfs(x,y):
            print(x,y)
            if x<0 or x>m-1 or y<0 or y>n-1:return False
            if grid[i][j] == -1:return
            if grid[i][j] == 1:return True
            grid[i][j] = -1
            for dx,dy in dirs:
                nx,ny = x+dx,y+dy
                if dfs(nx,ny) == False:return False
            return True
        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0 and dfs(i,j)==True:
                    print(i,j,"=================")
                    count = count+1
                    print(grid)
        return count

if __name__ == '__main__':
    solution = Solution()
    nums = [[0,0,1,1,0,1,0,0,1,0],[1,1,0,1,1,0,1,1,1,0],[1,0,1,1,1,0,0,1,1,0],[0,1,1,0,0,0,0,1,0,1],[0,0,0,0,0,0,1,1,1,0],[0,1,0,1,0,1,0,1,1,1],[1,0,1,0,1,1,0,0,0,1],[1,1,1,1,1,1,0,0,0,0],[1,1,1,0,0,1,0,1,0,1],[1,1,1,0,1,1,0,1,1,0]]
    solution.closedIsland(nums)