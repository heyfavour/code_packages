class Solution:
    def totalNQueens(self, n: int) -> int:
        ans = 0
        matrix = [[0]*n for i in range(n)]
        def check(matrix,i,j):
            def diagonal(i,j):
                side_1= [matrix[i+m][j+m] for m in range(min([n-i,n-j]))]
                side_2= [matrix[i-m][j-m] for m in range(min([i+1,j+1]))]
                side_3= [matrix[i-m][j+m] for m in range(min([i+1,n-j]))]
                side_4= [matrix[i+m][j-m] for m in range(min([n-i,j+1]))]
                side = side_1+side_2+side_3+side_4
                return side
            return sum(matrix[i]+ [matrix[row][j] for row in range(n)] +diagonal(i,j))

        def backtrack(matrix,row,queen_num):
            if queen_num == n:
                nonlocal ans
                ans = ans + 1
                return True
            for j in range(n):
                if check(matrix,row,j):continue
                matrix[row][j] = 1
                backtrack(matrix,row+1,queen_num+1)
                matrix[row][j] = 0
        backtrack(matrix,0,0)
        return ans

if __name__ == '__main__':
    n = 4
    solution = Solution()
    print(solution.totalNQueens(n))