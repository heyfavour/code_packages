from typing import List


class Solution:
    @classmethod
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        nlist = [i + 1 for i in range(target)]
        result = []
        for i in range(1, target + 1):
            sum = i
            j = i + 1
            while j <= target and sum < target:
                sum = sum + j
                if sum == target: result.append(nlist[i-1:j])
                j = j + 1
        return result


if __name__ == '__main__':
    print(Solution.findContinuousSequence(15))
