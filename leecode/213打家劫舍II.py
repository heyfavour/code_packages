from typing import List


class Solution:
    @classmethod
    def rob(self, nums: List[int]) -> int:
        dp_1 = [0] * (len(nums) - 1)
        nums_1 = nums[0:-1]
        for i in range(len(nums_1)):
            if i == 0:
                dp_1[0] = nums_1[i]
            elif i == 1:
                dp_1[i] = max(nums_1[0], nums_1[1])
            else:
                dp_1[i] = max(dp_1[i - 1], dp_1[i - 2] + nums_1[i])
        dp_2 = [0] * (len(nums) - 1)
        nums_2 = nums[1:]
        for i in range(len(nums_2)):
            if i == 0:
                dp_2[0] = nums_2[i]
            elif i == 1:
                dp_2[i] = max(nums_2[0], nums_2[1])
            else:
                dp_2[i] = max(dp_2[i - 1], dp_2[i - 2] + nums_2[i])

        return max((dp_1[-1], dp_2[-1]))


if __name__ == '__main__':
    nums = [2, 3, 2, 2, 4, 5]
    print(Solution.rob(nums))
