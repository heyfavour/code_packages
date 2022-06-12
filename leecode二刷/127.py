import string
from collections import defaultdict, deque


class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: list[str]) -> int:
        word_set = set(wordList)
        # 构建graph
        if endWord not in word_set: return 0
        graph = defaultdict(set)
        found = self._bfs(beginWord, endWord, word_set, graph)
        if not found: return 0
        # dfs查询最短
        path = self._dfs(beginWord, endWord, graph)
        return path

    def _dfs(self, beginWord, endWord, graph):
        path = 1

        def _dfs_help(start, end):
            nonlocal path
            if start == end: return path
            for word in graph[start]:
                path = path + 1
                ans = _dfs_help(word, end)
                if ans: return ans
                path = path - 1

        return _dfs_help(beginWord, endWord)

    def _bfs(self, start, end, wordset, graph):
        Q = deque()
        Q.append(start)

        visted = set()
        visted.add(start)

        next_visited = set()
        word_len = len(start)
        found = False

        while Q:
            for i in range(len(Q)):
                string_word = Q.popleft()
                word = list(string_word)
                for i in range(word_len):
                    old_char = word[i]
                    for char in string.ascii_lowercase:  # 替换字符
                        word[i] = char
                        new_word = "".join(word)
                        if new_word in wordset and new_word not in visted:  # 构建关系
                            graph[string_word].add(new_word)
                            if new_word not in next_visited:  # 没访问过则塞进队列 防止成功重复
                                Q.append(new_word)
                                next_visited.add(new_word)
                            if new_word == end: found = True
                    word[i] = old_char
            if found == True: break
            visted = visted | next_visited
        return found


if __name__ == '__main__':
    beginWord = "hit"
    endWord = "cog"
    wordList = ["hot", "dot", "dog", "lot", "log", "cog"]
    solution = Solution()
    print(solution.ladderLength(beginWord, endWord, wordList))
