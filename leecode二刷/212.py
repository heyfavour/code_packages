from collections import defaultdict


class Trie():
    def __init__(self):
        self.children = defaultdict(Trie)
        self.word = ""

    def insert(self, word):
        node = self
        for ch in word:
            node = node.children[ch]
        node.word = word


class Solution:
    def findWords(self, board: list[list[str]], words: list[str]) -> list[str]:
        trie = Trie()
        for word in words: trie.insert(word)

        m, n, ans = len(board), len(board[0]), set()
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def dfs(i, j, node):
            if not (0 <= i < m and 0 <= j < n) or not board[i][j] in node.children: return
            ch = board[i][j]
            node = node.children[ch]
            if node.word: ans.add(node.word)
            board[i][j] = "#"
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                dfs(ni, nj, node)
            board[i][j] = ch

        for i in range(m):
            for j in range(n):
                dfs(i, j, trie)
        return list(ans)


if __name__ == '__main__':
    board = [["o", "a", "b", "n"], ["o", "t", "a", "e"], ["a", "h", "k", "r"], ["a", "f", "l", "v"]]
    words = ["oa", "oaa"]
    # board = [["o", "a", "a", "n"], ["e", "t", "a", "e"], ["i", "h", "k", "r"], ["i", "f", "l", "v"]]
    # words = ["oath", "pea", "eat", "rain"]
    solution = Solution()
    print(solution.findWords(board, words))
