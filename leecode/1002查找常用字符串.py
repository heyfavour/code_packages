from typing import List

class Solution:
    def commonChars(self, A: List[str]) -> List[str]:

        def get_str_dict(string):
            string_dict = {}
            for s in string:
                string_dict[s] = string_dict.setdefault(s,0)  + 1
            return string_dict

        result = get_str_dict(A[0])
        for s in A[1:]:
            string = get_str_dict(s)
            for i in result:
                result[i] = min([string.get(i,0),result[i]])
        return [i for i,v in result.items() if v>0 for _ in range(v)]



if __name__ == '__main__':
    L = ["cool","loock","cook"]
    s = Solution()
    print(s.commonChars(L))
