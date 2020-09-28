from typing import List
import copy

class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        self.result = []
        dict_map = {0:".",1:"Q"}
        board = [[0 for i in range(n)] for i in range(n)]
        def back(i, board, n):
            #终止条件
            if i==n:
                self.result.append(["".join([dict_map[i] for i in row]) for row in board])
                return True
            for jp in range(n):
                board[i][jp] = 1
                if not self.check(i, jp, board, n):
                    board[i][jp] = 0
                    continue
                back(i+1,board,n)
                board[i][jp] = 0
            return False
        back(0,board, n)
        return self.result


    def check(self, i, j, board, n):
        def side(i,j,board,n):
            side_1111 = [(i-x,j+x) for x in range(0,min([i+1,n-j]))]
            side_0101 = [(i+x,j-x) for x in range(0,min([n-i,j+1]))]
            side_0111 = [(i-x,j-x) for x in range(0,min([i+1,j+1]))]
            side_1101 = [(i+x,j+x) for x in range(0,min([n-i,n-j]))]
            side = side_1111 + side_0101 +  side_0111 + side_1101
            return sum([board[i][j] for (i,j) in side])
        if sum(board[i]) > 1 or sum([board[row][j] for row in range(n)]) > 1 or side(i,j,board,n)>4:
            return False
        return True

if __name__ == '__main__':
    s = Solution()
    n = 4
    print(s.solveNQueens(n))
