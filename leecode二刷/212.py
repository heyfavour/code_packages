from collections import defaultdict


class Trie():
    def __init__(self):
        self.children = [None] * 26
        self.end = False

    def insert(self, word):
        node = self
        for ch in word:
            pos = ord(ch) - ord("a")
            if not node.children[pos]: node.children[pos] = Trie()
            node = node.children[pos]
        node.end = True
        node.word = word


class Solution:
    def findWords(self, board: list[list[str]], words: list[str]) -> list[str]:
        # 数据准备
        trie = Trie()
        for word in words: trie.insert(word)  # 将所有要查询的数据塞到一个前缀树
        # 基数准备
        m,n = len(words),len(words[0])
        ans = []
        dirs = [(-1, 0), (1, 0), (0, 1), (0 - 1)]

        def dfs(i, j, node):  # 从某个节点开始查询
            pos = ord(board[i][j]) - ord("a")
            if not node.children[pos]: return
            if node.end: ans.append(node.word)
            node = node.children[pos]
            for dx, dy in dirs:
                ni, nj = i + dx, j + dy
                ch = board[ni][nj]
                dfs(ni, nj, node)
                board[ni][nj] = ch

        for i in range(m):
            for j in range(n):
                dfs(i, j, trie)
        return ans


if __name__ == '__main__':
    board = [["o", "a", "b", "n"], ["o", "t", "a", "e"], ["a", "h", "k", "r"], ["a", "f", "l", "v"]]
    words = ["oa", "oaa"]
    solution = Solution()
    print(solution.findWords(board, words))
