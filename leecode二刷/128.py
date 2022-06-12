class Solution:
    def longestConsecutive(self, nums: list[int]) -> int:
        hash_dict = {}
        max_len = 0
        for num in nums:
            if num not in hash_dict:
                left = hash_dict.get(num - 1, 0)
                right = hash_dict.get(num + 1, 0)
                cur_len = right + left + 1
                max_len = max(cur_len, max_len)
                hash_dict[num] = cur_len
                hash_dict[num - left] = cur_len  # 边界值
                hash_dict[num + right] = cur_len
                print(hash_dict)
        return max_len


if __name__ == '__main__':
    nums = [0, 3, 7, 2, 5, 8, 4, 6, 0, 1]
    solution = Solution()
    print(solution.longestConsecutive(nums))
