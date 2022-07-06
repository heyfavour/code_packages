# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        #1
        # stack = []
        # count = 0
        # def dfs(node):
        #     nonlocal count
        #     if not node:return
        #     dfs(node.left)
        #     stack.append(node.val)
        #     count = count+1
        #     if count == k:raise Exception("OUT")
        #     dfs(node.right)
        # try:
        #     dfs(root)
        # except Exception as e:
        #     pass
        # return stack.pop()
        #2
        # def gen(node):
        #     if node:
        #         yield from gen(node.left)
        #         yield node.val
        #         yield from gen(node.right)
        # iter = gen(root)
        # for i in range(k):
        #     ans = next(iter)
        # return ans
        #3
        stack = []
        while root or stack:
            while root:
                stack.append(root)
                root=root.left
            root = stack.pop()
            k = k - 1
            if k ==0:return root.val
            root = root.right