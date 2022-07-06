class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m ,n = len(matrix),len(matrix[0])
        i,j = 0,n-1
        while 0<=i<m and 0<=j<n:
            if matrix[i][j]==target:
                return True
            elif matrix[i][j]>target:
                j = j-1
            elif matrix[i][j]<target:
                i = i+1
        return False