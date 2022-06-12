class Solution:
    def majorityElement(self, nums: list[int]) -> int:
        count = 0
        for num in nums:
            if count == 0: candicate = num
            count += 1 if num == candicate else -1
        return candicate


if __name__ == '__main__':
    solution = Solution()
    nums = [2, 2, 1, 1, 1, 2, 2]
    print(solution.majorityElement(nums))
