class Solution:
    def maximalSquare(self, matrix: list[list[str]]) -> int:
        m, n = len(matrix), len(matrix[0])
        # min_len = min(m,n)
        # dp = [set() for _ in range(min_len+1)]
        # dp[0]= set((i,j) for i in range(m) for j in range(n) if matrix[i][j]== "1")#1 坐标
        # if len(dp[0]) == m*n:return min_len**2
        # if min_len == 1 and dp[0]:return 1
        # if not dp[0]:return 0
        # for i in range(1,min(m,n)+1):
        #     for x,y in dp[i-1]:
        #         if x+i >=m or y+i>=n:continue
        #         if ((x+1,y) in dp[i-1]) and (x,y+1) in dp[i-1] and (x+1,y+1) in dp[i-1]:
        #             dp[i].add((x,y))
        #     if dp[i] == set():return i**2
        dp = [[0] * n for _ in range(m)]
        max_len = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == "1":
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
                    max_len = max(max_len, dp[i][j])
        return max_len ** 2


if __name__ == '__main__':
    solution = Solution()
    matrix = [["1", "0", "1", "0", "0"], ["1", "0", "1", "1", "1"], ["1", "1", "1", "1", "1"],
              ["1", "0", "0", "1", "0"]]
    print(solution.maximalSquare(matrix))
