class Solution:
    def dfs(self,node,num):
        if not node:return 1
        if node.val !=num:return 0
        return self.dfs(node.left,num) and self.dfs(node.right,num)

    def countUnivalSubtrees(self, root: Optional[TreeNode]) -> int:
        if root == None:return 0
        return self.dfs(root,root.val)+self.countUnivalSubtrees(root.left)+self.countUnivalSubtrees(root.right)