class Solution:
    def search(self, nums: list[int], target: int) -> int:
        """
        n+1 ...2n 0  1 。。。n
        """
        n = len(nums)
        L, R = 0, n - 1
        while L <= R:
            mid = (L + R) >> 1
            if nums[mid] == target:
                return mid
            elif nums[0] <= nums[mid]:  # 左边有序
                if nums[0] <= target < nums[mid]:  # 左边有序且落在左边
                    R = mid - 1  # R = mid - 1
                else:
                    L = mid + 1  # L 左边有序但落在右边L=mid+1
            else:  # 右边有序
                if nums[mid] < target <= nums[R]:  # 右边有序且落在右边
                    L = mid + 1
                else:
                    R = mid - 1
        return -1


if __name__ == '__main__':
    nums = [4, 5, 6, 7, 0, 1, 2]
    target = 0
    solution = Solution()
    print(solution.search(nums, target))
