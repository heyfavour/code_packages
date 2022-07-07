class Solution:
    def shortestDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        pos1 ,pos2 = None,None
        ans = float("inf")
        for i in range(len(wordsDict)):
            if wordsDict[i] == word1:
                pos1=i
                if pos1 !=None and pos2!=None:ans = min(ans,abs(pos1-pos2))
            if wordsDict[i] == word2:
                pos2=i
                if pos1 !=None and pos2!=None:ans = min(ans,abs(pos1-pos2))
        return ans