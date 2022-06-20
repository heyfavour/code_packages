class Solution:
    def reverseWords(self, s: list[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        def swap(left,right):
            while left<=right:
                s[left],s[right] = s[right],s[left]
                left = left+1
                right=right-1
        left = right = 0
        for right in range(1,len(s)):
            if s[right] == " ":
                swap(left,right-1)
                left = right+1
        swap(left,right)
        swap(0,len(s)-1)


if __name__ == '__main__':
    s = ["t", "h", "e", " ", "s", "k", "y", " ", "i", "s", " ", "b", "l", "u", "e"]
    solution = Solution()
    print(solution.reverseWords(s))
