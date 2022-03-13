# 旋转矩阵
def rotate_matrix():
    matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    matrix = list(zip(*matrix))[::-1]  # 逆时针
    matrix = list(zip(*matrix[::-1]))  # 顺时针
