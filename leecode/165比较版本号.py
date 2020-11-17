class Solution:
    @classmethod
    def compareVersion(self, version1: str, version2: str) -> int:
        version1 = version1.split(".")
        version2 = version2.split(".")
        for i in range(max([len(version1), len(version2)])):
            v1 = int(version1[i]) if i < len(version1) else 0
            v2 = int(version2[i]) if i < len(version2) else 0
            if v1 > v2:
                return 1
            elif v1 < v2:
                return -1


if __name__ == '__main__':
    version1 = "7.5.2.4"
    version2 = "7.5.3"
    print(Solution.compareVersion(version1, version2))
