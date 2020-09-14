from typing import List


class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        visited = set()
        x = len(board)
        y = len(board[0])

        def search(i, j, k):
            # 判断条件
            # 当前条件
            if board[i][j] != word[k]:
                return False
            # 终止条件
            if k == len(word) - 1:
                return True

            # 遍历
            visited.add((i, j))
            for di, dj in directions:
                newi, newj = i + di, j + dj
                if (newi, newj) not in visited and 0 <= newi < x and 0 <= newj < y:
                    if search(newi, newj, k + 1):
                        return True
            visited.remove((i, j))

        for i in range(x):
            for j in range(y):
                if (i, j) != (1, 3): continue
                if search(i, j, 0):
                    return True
        return False


if __name__ == '__main__':
    board = [
        ['A', 'B', 'C', 'E'],
        ['S', 'F', 'C', 'S'],
        ['A', 'D', 'E', 'E']
    ]
    s = Solution()
    w = "SEE"
    print(s.exist(board, w))
