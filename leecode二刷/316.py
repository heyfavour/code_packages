import collections
class Solution:
    def removeDuplicateLetters(self, s) -> int:
        stack = []
        seen = set()
        remain_counter = collections.Counter(s)

        for c in s:
            if c not in seen:
                while stack and c < stack[-1] and  remain_counter[stack[-1]] > 0:
                    seen.remove(stack.pop())
                seen.add(c)
                stack.append(c)
            remain_counter[c] -= 1
        return ''.join(stack)



if __name__ == '__main__':
    solution = Solution()
    solution.removeDuplicateLetters("cbacdcbc")