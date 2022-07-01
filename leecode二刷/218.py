import heapq


class Solution:
    def getSkyline(self, buildings: list[list[int]]) -> list[list[int]]:
        # 超时
        # L = [left for left, _, _ in buildings] + [right for _, right, _ in buildings]
        # L = sorted(list(set(L)))
        # L = [[i, [0]] for i in L]
        # i = 0
        # for left, right, height in buildings:
        #     for j in range(i, len(L)):
        #         if left == L[j][0]: i = j
        #         if right < L[j][0]: break
        #         if left <= L[j][0] < right: L[j][1].append(height)
        # L = [[i[0], max(i[1])] for i in L]
        # ans = []
        # for i in range(len(L)):
        #     if i>0 and L[i][1] == L[i-1][1]:continue
        #     ans.append(L[i])
        # return ans
        L = [(l,-h,r) for l, r, h in buildings] + [(r,0,0) for _, r, _ in buildings]#左端点 高的排前面 + 右端点
        L.sort()
        ans = []#(L,H) #插入 00 检测第一次变化
        heapq_stack = [(0,float("inf"))]#(H,R)用于记录[L,R]有效高度 最大堆 随时取最高 初始化保证0一直在 这样遇到重复的r重点
        for left,minusH,right in L:
            H=-minusH
            while left>=heapq_stack[0][1]:
                heapq.heappop(heapq_stack)
            if H:heapq.heappush(heapq_stack,(-H,right))
            if not ans or ans[-1][1]!=-heapq_stack[0][0]:#第一次  最高发生变化了#(L,H) (-H R)
                ans.append((left,-heapq_stack[0][0]))
        return ans


if __name__ == '__main__':
    solution = Solution()
    buildings = [[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]]
    print(solution.getSkyline(buildings))
