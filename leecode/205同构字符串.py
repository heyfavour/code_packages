class Solution:
    @classmethod
    def isIsomorphic(self, s: str, t: str) -> bool:
        map_dict = {}
        for i,k in enumerate(s):
            if map_dict.get(k) is not None and map_dict[k] != t[i]:
                return False
            map_dict[k] = t[i]
        print(map_dict)
        if set(map_dict.keys())!=set(map_dict.values()):return  False
        return  True



if __name__ == '__main__':
    s = "egg"
    t = "add"
    print(Solution.isIsomorphic(s,t))
