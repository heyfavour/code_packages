from collections import defaultdict
class Solution:
    def groupStrings(self, strings: list[str]) -> list[list[str]]:
        _group = defaultdict(list)
        for i in strings:
            group = []
            j0 = ord(i[0])
            for j in i:
                num = ord(j)-j0
                if num<0:num = num+26
                group.append(str(num))
            _group["|".join(group)].append(i)
        L = list(_group.values())
        return sorted(L,key=lambda x:len(x[0]),reverse=True)
if __name__ == '__main__':
    solution = Solution()
    strings = ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"]
    print(solution.groupStrings(strings))