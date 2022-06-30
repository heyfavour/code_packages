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
        #优化1
        L = [left for left, _, _ in buildings] + [right for _, right, _ in buildings]
        L = sorted(list(set(L)))
        L = [[i, [0]] for i in L]
        stack = []
        heapq.heapify(stack)
        build_index = 0
        for i in range(len(L)):
            if buildings[build_index]<=
        L = [[i[0], max(i[1])] for i in L]
        ans = []
        for i in range(len(L)):
            if i>0 and L[i][1] == L[i-1][1]:continue
            ans.append(L[i])
        return ans


    # now_building = []
    # for x in range(L):
    #     #获取l<=x<=r的max_H


if __name__ == '__main__':
    solution = Solution()
    buildings = [[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]]
    print(solution.getSkyline(buildings))
