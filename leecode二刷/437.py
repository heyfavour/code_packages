# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        # ans = 0
        # def check(node,target):
        #     nonlocal ans
        #     if node is None:return 0
        #     if node.val == target:ans = ans + 1
        #     check(node.left,target-node.val)
        #     check(node.right,target-node.val)

        # def dfs(node):
        #     if not node:return 0
        #     check(node,targetSum)
        #     dfs(node.left)
        #     dfs(node.right)
        # dfs(root)
        # return ans
        ans = 0
        prefix = {0:1}
        def dfs_prefix(node,prefix_sum):
            nonlocal ans
            if not node:return
            prefix_sum = prefix_sum + node.val
            ans = ans + prefix.get(prefix_sum-targetSum,0)
            prefix[prefix_sum] = prefix.get(prefix_sum,0)+1
            dfs_prefix(node.left,prefix_sum)
            dfs_prefix(node.right,prefix_sum)
            prefix[prefix_sum] = prefix.get(prefix_sum,0)-1
        dfs_prefix(root,0)
        return ans