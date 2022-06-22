class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        """
        count = 0
        while left<right:
            left = left>>1
            right = right>>1
            count=count+1
        return left<<count
        """
        while left<right:
            right=right&(right-1)
        return right


if __name__ == '__main__':
    solution  = Solution()
    left = 5
    right = 7
    print(solution.rangeBitwiseAnd(left,right))