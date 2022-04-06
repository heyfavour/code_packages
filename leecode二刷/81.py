class Solution:
    def search(self, nums: list[int], target: int) -> bool:
        L, R = 0, len(nums) - 1
        while L <= R:
            mid = (L + R) >> 1
            if nums[mid] == target:
                return True
            elif nums[L] == nums[mid] == nums[R]:
                L = L + 1
                R = R - 1
            elif nums[L] <= nums[mid]:  # 左边有序
                if nums[L] <= target <= nums[mid]:  # 左边有序且落在左边
                    R = mid - 1
                else:
                    L = mid + 1
            else:  # 右边有序
                if nums[mid] < target <= nums[R]:
                    L = mid + 1
                else:
                    R = mid - 1
        return False


if __name__ == '__main__':
    slution = Solution()
    nums = [2, 5, 6, 0, 0, 1, 2]
    nums = [1, 0, 1, 1, 1]
    target = 0
    solution = Solution()
    print(solution.search(nums, target))
