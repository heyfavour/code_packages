class Solution:
    def nextPermutation(self, nums: list[int]) -> None:
        """
        5[4]65321
        5[4]6[5]321
        5[5]6[4]321
        5 5 6 4 3 2 1
        5 5 1 2 3 4 6
        1234 5
        9876

        1234
        8765
        """
        n = len(nums)
        break_falg = False
        for i in range(n - 2, -1, -1):
            if nums[i] < nums[i + 1]:
                for j in range(n - 1, i, -1):
                    if nums[j] > nums[i]:
                        nums[i], nums[j] = nums[j], nums[i]
                        break_falg = True
                        break
            if break_falg: break
        if break_falg: i = i + 1
        L = i
        R = n - 1
        while L < R:
            nums[L], nums[R] = nums[R], nums[L]
            L = L + 1
            R = R - 1


if __name__ == '__main__':
    solution = Solution()
    nums = [5, 4, 6, 5, 3, 2, 1]
    # nums =[1,2,3]
    nums = [3, 2, 1]
    # nums = [1,3,2] # 1 3 2  [1] 3 [2]  2 3 1  2 1 3
    print(solution.nextPermutation(nums))
