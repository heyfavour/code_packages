from typing import List


class Solution:
    @classmethod
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        for i in range(len(numbers)):
            tmp = target - numbers[i]
            l, r = i + 1, len(numbers) - 1
            while l <= r:
                mid = (l + r) // 2
                if numbers[mid] == tmp:
                    return [i, mid]
                elif numbers[mid] > tmp:
                    r = mid - 1
                elif numbers[mid] < tmp:
                    l = mid + 1


if __name__ == '__main__': ''
numbers = [2, 7, 11, 15]
target = 9
print(Solution.twoSum(numbers, target))
