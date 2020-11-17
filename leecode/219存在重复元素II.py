from typing import List


class Solution:
    @classmethod
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        nums_dict = {}
        for i in range(len(nums)):
            nums_dict[nums[i]] = nums_dict.get(nums[i], 0) + 1
            if nums_dict[nums[i]]>1:return True
            if i>=k:nums_dict[nums[i-k]] = nums_dict.get(nums[i-k],0) - 1
        return False




if __name__ == '__main__':
    nums = [1,2,3,1,2,3]
    k = 3
    print(Solution.containsNearbyDuplicate(nums,k))
