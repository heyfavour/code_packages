"""
贼慢
class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.val = 0
        self.lazy = 0

class SegTree:
    def __init__(self,size):
        self.size = size
        self.root = Node()

    #下推 动态开点
    def pushdown(self,node,left,right):
        if not node.left:node.left = Node()
        if not node.right:node.right = Node()
        #未被标记过
        if node.lazy == 0:return
        #更新下推值
        node.left.val = node.left.val + node.lazy*left
        node.right.val = node.right.val + node.lazy*right
        #标记下推
        node.left.lazy = node.left.lazy + node.lazy
        node.right.lazy = node.right.lazy + node.lazy
        node.lazy = 0


    def pushup(self,node):
        node.val = node.left.val+node.right.val

    def update(self,node,start,end,L,R,add):
        #满足
        if L<=start<=end<=R:
            #node [start,end] 更新
            node.val = node.val + (end-start+1)*add
            node.lazy = node.lazy + add
            return
        #不满足 下推
        mid = (start+end)>>1
        self.pushdown(node,mid-start+1,end-mid)
        if L<=mid:self.update(node.left,start,mid,L,R,add)
        if R>mid:self.update(node.right,mid+1,end,L,R,add)
        #更新节点
        self.pushup(node)

    def query(self,node,start,end,L,R):
        if L<=start<=end<=R:return node.val
        mid = (start+end)>>1
        self.pushdown(node,mid-start+1,end-mid)
        ans = 0
        if L<=mid:ans = ans+self.query(node.left,start,mid,L,R)
        if R>mid:ans = ans+self.query(node.right,mid+1,end,L,R)
        return ans

class MyCalendar:
    def __init__(self):
        self.size = 10**9
        self.seg_tree = SegTree(size=self.size)

    def book(self, start: int, end: int) -> bool:
        if self.seg_tree.query(self.seg_tree.root, 0, self.size, start, end-1) != 0:return False
        self.seg_tree.update(self.seg_tree.root, 0, self.size, start, end-1, 1)
        return True
"""

class MyCalendar:
    def __init__(self):
        self.booked = []

    def book(self, start: int, end: int) -> bool:
        if any(l < end and start < r for l, r in self.booked):#[l,r) [start,end)  如果补充和 l>=end or start>=r ->  l<end and start<r
            return False
        self.booked.append((start, end))
        return True



if __name__ == '__main__':
    # [10, 20], [15, 25], [20, 30]]
    calendar = MyCalendar()
    print(calendar.book(10, 20))
    print(calendar.book(15, 25))
    print(calendar.book(20, 30))
