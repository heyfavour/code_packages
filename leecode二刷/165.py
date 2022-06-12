class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        version1 = version1.split(".")
        version2 = version2.split(".")
        v1_len = len(version1)
        v2_len = len(version2)
        print(version1, version2)
        if v1_len < v2_len:
            version1 = version1 + ["0"] * (v2_len - v1_len)
        else:
            version2 = version2 + ["0"] * (v1_len - v2_len)
        for i in range(max(v1_len,v2_len)):
            if int(version1[i])>int(version2[i]):return 1
            if int(version1[i])<int(version2[i]):return -1
        return 0


if __name__ == '__main__':
    solution = Solution()
    version1 = "0.1"
    version2 = "1.1"
    print(solution.compareVersion(version1, version2))
