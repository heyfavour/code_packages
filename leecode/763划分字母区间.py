from typing import List
import time


class Solution:
    @classmethod
    def partitionLabels(self, S: str) -> List[int]:
        i, j, start, p = 0, len(S) - 1, 0, 0
        result = []
        print(len(S))
        while i < len(S):
            print(f"i=={i}:{S[i]},j=={j}:{S[j]},p=={p}:{S[p]},{S[start:p + 1]},{result}")
            if i < j and S[i] != S[j] and j > p:
                j = j - 1
            elif i < j and S[i] == S[j]:
                i = i + 1
                p = j
                j = len(S) - 1
            elif i < j and j == p:
                i = i + 1
                j = len(S) - 1
            elif i == p:
                result.append(S[start:p + 1])
                start = p + 1
                i = i + 1
                j = len(S) - 1
                p = 0
            else:
                p = j
        print(result)

if __name__ == '__main__':
    S = "ababcbacadefegdehijhklij"
    S = "eaaaabaaecmno"
    print(Solution.partitionLabels(S))
