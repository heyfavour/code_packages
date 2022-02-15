"""
二分模板
n = len(nums)
left,right = 0,n-1
while left <=right:
    mid = (left + right) >> 1
    if nums[mid] == target:
        return mid
    elif nums[mid] < target:
        left = mid + 1
    elif num[mid] > target:
        right = mid -1

left,right = 0,n-1 => while left<=right =>left=right+1 [right+1 right]
left,right = 0,n   => while left<right  =>left=right
#左侧边界二分 [left right)
left,right = 0,n #注意
while left<right:#注意
    mid = (left + right) >> 1
    if nums[mid] == target:
        right = mid #注意
    elif nums[mid] < target:
        left = mid + 1
    elif nums[mid] > target:
        right = mid #注意
return left
#右侧边界二分 (left,right]
left,right= 0,n
while left <=right:
    if nums[left]==target:
        left = mid + 1 #注意
    elif nums[mid]<target:
        left = mid + 1
    elif nums[mid]>target:
        right = mid
return left - 1   #注意

"""
###
#分析
# 0 1 2 3 4 5 6 7 8 9 10 11 12
# 12 是单个
# 12 + 0 >> 1 = 6 6-7 偶数 偶数和下一个相等 收缩左边界 left = mid + 1 = 7

###
class Solution:
    #异或运算 不是很能理解
    # def singleNonDuplicate(self, nums: list[int]) -> int:
    #     left, right = 0, len(nums) - 1
    #     while left < right:
    #         mid = (left + right) >> 1
    #         if nums[left] == nums[left^1]:
    #             left = mid + 1
    #         else:
    #             right = mid
    #     return nums[left]

    def singleNonDuplicate(self, nums: list[int]) -> int:
        left,right = 0,len(nums) - 1
        #单一元素出现在索引为偶数的位置
        while left<right:#终止于left=right [left,right）
            mid = (left + right) >>1
            if (mid%2==0):#偶数
                if nums[mid] == nums[mid+1]:
                    left = mid + 1
                else:
                    right = mid
            else:
                if nums[mid] == nums[mid- 1]:
                    left = mid + 1
                else:
                    right = mid
        return nums[left]
if __name__ == '__main__':
    solution = Solution()
    nums = [3, 3, 7, 7, 10, 11, 11]
    num = solution.singleNonDuplicate(nums)
    print(num)
