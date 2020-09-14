from typing import List
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def averageOfLevels(self, root: TreeNode) -> List[float]:

        result = {}

        def get_node_list(root, node_level):
            result.setdefault(node_level, [0, 0])
            result[node_level][0] = result[node_level][0] + root.val
            result[node_level][1] = result[node_level][1] + 1
            if root.left:
                get_node_list(root.left, node_level + 1)
            if root.right:
                get_node_list(root.right, node_level + 1)

        get_node_list(root, 0)
        return [v[0] / v[1] for k, v in result.items()]
