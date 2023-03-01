class Node:
    def __init__(self, left=None, right=None, val=0, lazy=0):
        """
        left:   左孩子
        right:  右孩子
        val:    值
        lazy:   懒惰标记，是0说明没有懒惰标记，是正数说明这里的懒惰标记还未下放
        """
        self.left = left
        self.right = right
        self.val = val
        self.lazy = lazy


class MySegTree:
    def __init__(self, size):
        """
        size:   线段树的总大小（根节点管理的区间的长度）
        """
        self.size = size
        self.root = Node()

    def push_down(self, node, s, e):
        """
        向下更新，并传递懒惰更新标志
        node:   当前节点
        s:      start，当前节点管理的左边界（含）
        e:      end，当前节点管理的右边界（含）
        """
        mid = s + ((e - s) >> 1)
        if node.left is None:
            node.left = Node()
        if node.right is None:
            node.right = Node()
        if node.lazy == 0:
            return
        node.left.val += node.lazy * (mid - s + 1)
        node.right.val += node.lazy * (e - mid)
        node.left.lazy += node.lazy
        node.right.lazy += node.lazy
        node.lazy = 0
        return

    def push_up(self, node):
        """
        向上更新，要求node的两个子节点均已更新完
        node:   当前节点
        """
        node.val = node.left.val + node.right.val
        return

    def update(self, node, s, e, l, r, add):
        """
        更新闭区间[l, r]，给此区间内的每个值，都加上add
        闭区间[l, r]和当前节点管理的区间[s, e]的交集一定是非空的
        node:   当前节点
        s:      start，当前节点管理的左边界（含）
        e:      end，当前节点管理的右边界（含）
        l:      left，要更改的区间的左边界
        r:      right，要更改的区间的右边界
        add:    addition，增量
        """
        if l <= s and e <= r:
            node.val += add * (e - s + 1)
            node.lazy += add
            return

        self.push_down(node, s, e)
        mid = s + ((e - s) >> 1)

        if l <= mid:
            self.update(node.left, s, mid, l, r, add)
        if r > mid:
            self.update(node.right, mid + 1, e, l, r, add)

        self.push_up(node)
        return

    def query(self, node, s, e, l, r):
        """
        查询闭区间[l, r]的值
        闭区间[l, r]和当前区间[s, e]一定是有交集的
        node:   当前节点
        s:      start，当前节点管理的左边界（含）
        e:      end，当前节点管理的右边界（含）
        l:      left，要更改的区间的左边界
        r:      right，要更改的区间的右边界
        """
        if l <= s and e <= r:
            return node.val


        self.push_down(node, s, e)
        mid = s + ((e - s) >> 1)

        ans = 0
        if l <= mid:
            ans += self.query(node.left, s, mid, l, r)
        if r > mid:
            ans += self.query(node.right, mid + 1, e, l, r)
        return ans

class MyCalendar:

    def __init__(self):
        self.size = 10**9
        self.seg_tree = MySegTree(size=self.size)

    def book(self, start: int, end: int) -> bool:
        if self.seg_tree.query(self.seg_tree.root, 0, self.size, start, end-1) != 0:
            return False
        self.seg_tree.update(self.seg_tree.root, 0, self.size, start, end-1, 1)
        return True

if __name__ == '__main__':
    calendar = MyCalendar()
    nums = [[10, 20], [15, 25], [20, 30]]
    for i in nums:
        print(calendar.book(i[0],i[1]))
        break