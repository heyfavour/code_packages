class Solution:
    def shortestWordDistance(self, wordsDict: list[str], word1: str, word2: str) -> int:
        pos_1,pos_2 = -1,-1
        ans = float("inf")
        if word1!=word2:
            for i in range(len(wordsDict)):
                if word1 == wordsDict[i]:
                    pos_1 = i
                    if pos_1>=0 and pos_2>=0:ans = min(ans,pos_1-pos_2)
                if word2 == wordsDict[i]:
                    pos_2 = i
                    if pos_1>=0 and pos_2>=0:ans = min(ans,pos_2-pos_1)
        else:
            last_pre = -1
            for i in range(len(wordsDict)):
                if word1 == wordsDict[i]:
                    if last_pre>=0:ans  = min(ans,i-last_pre)
                    last_pre = i
        return ans