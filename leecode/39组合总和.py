# def backtrack():
#     resulit = []
#     if (回朔点）：  # 这条路走到底的条件。也是递归出口
#         result.add(path)
#         return
#     for route in all_route_set:  逐步选择当前节点下的所有可能route
#         if 剪枝条件：
#             剪枝前的操作
#             return  # 不继续往下走了，退回上层，换个路再走
#         else ：  # 当前路径可能是条可行路径
#             保存当前数据  # 向下走之前要记住已经走过这个节点了。例如push当前节点
#             self.backtrack()  # 递归发生，继续向下走一步了。
#             回朔清理  # 该节点下的所有路径都走完了，清理堆栈，准备下一个递归。例如弹出当前节点

class Solution(object):
    # 初始版本 其实是递归 所有都遍历了一遍 只不过加加上了剪纸


    # 因为从小到大无限重复  所有小的已经把大的都遍历了，可以再次剪枝
    # 类似223  322 新增 新增tmp_i 每次只从>=tmp_i的地方开始遍历
    def combinationSum2(self, candidates, target):
        result = []

        def back(position,tmp):
            if sum(tmp) == target:
                result.append(tmp)

            for i in range(position,len(candidates)):
                if sum(tmp) + candidates[i] > target: break
                back(i,tmp + [candidates[i], ])

        back(0,[])
        return result


if __name__ == '__main__':
    s = Solution()
    l = [2, 3, 6, 7]

    print(s.combinationSum1(l, 7))
    print(s.combinationSum2(l, 7))
