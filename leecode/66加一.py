from typing import List
class Solution:
    @classmethod
    def plusOne(self, digits: List[int]) -> List[int]:
        digits.reverse()
        for i in range(len(digits)):
            digits[i] = digits[i] + 1
            if digits[i] >= 10:
                digits[i] = 0
                if i == len(digits)-1:digits.append(1)
            else:
                break
        digits.reverse()
        return digits

if __name__ == '__main__':
    num = [4, 9, 9, 9]
    print(Solution.plusOne(num))
