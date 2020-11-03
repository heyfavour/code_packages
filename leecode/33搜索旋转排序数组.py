from typing import List


class Solution:
    @classmethod
    def search(self, nums: List[int], target: int) -> int:

        def helper(num_list, l, r):
            if l <= r:
                mid = (l + r) // 2
                if num_list[mid] == target:
                    return mid
                elif num_list[mid] < nums[r]:
                    left =  helper(num_list, l, mid - 1)
                    if left>=0:return left
                    nl, nr = mid + 1, r
                    while nl <= nr:
                        mid = (nl + nr) // 2
                        if num_list[mid] == target:
                            return mid
                        elif num_list[mid] < target:
                            nl = mid + 1
                        elif num_list[mid] > target:
                            nr = mid - 1
                else:
                    nl, nr = l, mid - 1
                    right =  helper(num_list, mid + 1, r)
                    if right>=0:return right
                    while nl <= nr:
                        mid = (nl + nr) // 2
                        if num_list[mid] == target:
                            return mid
                        elif num_list[mid] < target:
                            nl = mid + 1
                        elif num_list[mid] > target:
                            nr = mid - 1

            return -1

        return helper(nums, 0, len(nums) - 1)


if __name__ == '__main__':
    nums = [4, 5, 6, 7, 8, 9, 10, 0, 1, 2]
    nums = [1,3,5,7,8]
    target = 1
    print(Solution.search(nums, target))
