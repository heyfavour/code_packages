class Solution:
    @classmethod
    def compressString(self, S: str) -> str:
        NS = ''
        i = 0
        while i <= len(S)-1:
            l,r  = i,i
            while S[l] == S[r]:
                r = r + 1
                if r>len(S) - 1:break
            NS = NS + S[l] + str(r - l)
            i = r
        return NS

if __name__ == '__main__':
    S = "aabcccccaaa"
    S = "abbccd"
    print(Solution.compressString(S))
