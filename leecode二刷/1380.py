class Solution:
    def luckyNumbers(self, matrix: list[list[int]]) -> list[int]:
        max_col = [max([matrix[row][col] for row in range(len(matrix))]) for col in range(len(matrix[0]))]
        min_row = [min(row) for row in matrix]
        lucy_list = []
        for i,row in enumerate(matrix):
            for j,col in enumerate(row):
                if matrix[i][j] == min_row[i] ==max_col[j]:lucy_list.append(matrix[i][j])
        return lucy_list



if __name__ == '__main__':
    matrix = [[3, 7, 8], [9, 11, 13], [15, 16, 17]]
    solution  = Solution()
    lucy_list = solution.luckyNumbers(matrix)
    print(lucy_list)
    print(list(zip(*matrix)))
