from typing import List


def gcd(a, b):
    return a if b == 0 else gcd(b, a % b)

class Solution:
    def simplifiedFractions(self, n: int) -> List[str]:
        return [f"{numerator}/{denominator}" for denominator in range(2, n + 1) for numerator in
                range(1, denominator + 1) if gcd(numerator,denominator) == 1]




if __name__ == '__main__':
    solution = Solution()
    asnswer = solution.simplifiedFractions(4)
    print(asnswer)
