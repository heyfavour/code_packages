import string
from collections import deque
from collections import defaultdict


class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: list[str]) -> list[list[str]]:
        word_set = set(wordList)
        if len(word_set) == 0 or endWord not in word_set: return []

        tree = defaultdict(set)

        found = self._bfs(beginWord, endWord, word_set, tree)  # 构建树形结构
        if not found: return []

        # 存在路径
        path = self._dfs(beginWord, endWord, tree)
        return path

    def _dfs(self, beginWord, endWord, tree):
        ans = []
        path = [beginWord, ]

        def help(start, end):
            if start == end: ans.append(path.copy())
            for word in tree[start]:
                path.append(word)
                help(word, end)
                path.pop()

        help(beginWord, endWord)
        return ans

    def _bfs(self, beginWord, endWord, word_set, tree):
        Q = deque()
        Q.append(beginWord)

        visited = set()
        visited.add(beginWord)

        found = False

        word_len = len(beginWord)
        next_visited = set()  # 本次访问的节点

        while Q:
            for i in range(len(Q)):
                string_word = Q.popleft()
                word = list(string_word)
                for j in range(word_len):
                    char = word[j]  # 记录原字符
                    for nc in string.ascii_lowercase:  # 更改字符并且查询
                        word[j] = nc
                        new_word = "".join(word)
                        if new_word in word_set and new_word not in visited:  # 新word在集合中并且没有构建节点
                            if new_word == endWord: found = True
                            if new_word not in next_visited:  # 本次没有访问锅
                                next_visited.add(new_word)  # 本次访问过
                                Q.append(new_word)  # 下次队列从这个节点出发访问周围节点
                            tree[string_word].add(new_word)  # 构建节点图 {node:(node node node)}
                    word[j] = char  # 复原word
            if found: break
            visited = visited | next_visited  # 合并集合
            next_visited.clear()
        return found


if __name__ == '__main__':
    solution = Solution()
    beginWord = "hit"
    endWord = "cog"
    wordList = ["hot", "dot", "dog", "lot", "log", "cog"]
    print(solution.findLadders(beginWord, endWord, wordList))
