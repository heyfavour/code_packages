class Solution:
    def maximumGap(self, nums: list[int]) -> int:
        res = 0

        # 通排序
        def bucket_sort(nums):
            max_len = len(str(max(nums)))
            for i in range(max_len):
                bucket = [[] for _ in range(10)]
                for num in nums: bucket[(num // (10 ** i)) % 10].append(num)
                nums = [num for b in bucket for num in b]
            return nums

        nums = bucket_sort(nums)
        for i in range(1, len(nums)):
            res = max(res, nums[i] - nums[i - 1])
        return res


if __name__ == '__main__':
    nums = [3, 6, 9, 1]
    solution = Solution()
    print(solution.maximumGap(nums))
