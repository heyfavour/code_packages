class Solution:
    def findUnsortedSubarray(self, nums: list[int]) -> int:
        n = len(nums)
        if n <= 1: return 0
        # L 最左侧 找到一个树 比右边所有的数都小
        rightmin = float("inf")
        l = n - 1
        for i in range(n - 1, -1, -1):
            if nums[i] > rightmin:
                l = i - 1
            rightmin = min(nums[i], rightmin)
        # R 最右侧 找到一个比左边都大的树
        leftmax = float("-inf")
        r = 0
        for i in range(n):
            if nums[i] < leftmax: r = i + 1
            leftmax = max(leftmax, nums[i])
        return r - l - 1 if r > l else 0


if __name__ == '__main__':
    solution = Solution()
    print(solution.findUnsortedSubarray(nums=[2, 6, 4, 8, 10, 9, 15]))
    # print(solution.findUnsortedSubarray(nums=[2, 1]))
