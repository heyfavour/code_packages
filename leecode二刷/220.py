class Solution:
    def containsNearbyAlmostDuplicate(self, nums: list[int], k: int, t: int) -> bool:
        # if not k:return False
        # # k 滑动窗口
        # def search(num,L):#L k
        #     left,right = 0,len(L)-1
        #     while left<=right:
        #         mid = (left+right)>>1
        #         if L[mid] == num:
        #             return mid,True
        #         elif L[mid]>num:
        #             right = mid - 1
        #         else:
        #             left = mid + 1
        #     return left,False
        #
        # L  = []
        # for i in range(len(nums)):
        #     if i == 0:
        #         L.append(nums[i])
        #     else:
        #         pos,flag = search(nums[i],L)
        #         if flag:return True
        #         if pos == 0:
        #             if abs(nums[i]-L[0])<=t :return True
        #         elif pos == len(L) :
        #             if abs(nums[i]-L[-1])<=t :return True
        #         else:
        #             if abs(nums[i]-L[pos-1])<=t or abs(nums[i]-L[pos])<=t:return True
        #         L.insert(pos,nums[i])#入窗口
        #         if i>=k:L.remove(nums[i-k])#出窗口
        # return False
        if not k: return False
        bucket_size = t + 1  # 1 2 t=1 bucket_size = 2
        bucket = {}  # 由于是统计桶不需要排序

        def get_bucket_id(num):
            id = (num // bucket_size) if num >= 0 else (((num + 1) // bucket_size) - 1)
            return id

        for i in range(len(nums)):
            bucket_id = get_bucket_id(nums[i])
            if bucket.get(bucket_id) is not None: return True
            bucket[bucket_id] = nums[i]
            if bucket.get(bucket_id - 1) and nums[i] - bucket.get(bucket_id - 1) <= t: return True
            if bucket.get(bucket_id + 1) and  bucket.get(bucket_id + 1) -nums[i] <= t: return True
            if i >= k:
                remove_id = get_bucket_id(nums[i - k])
                bucket.pop(remove_id)
        return False


if __name__ == '__main__':
    solution = Solution()
    nums = [1, 5, 9, 1, 5, 9]
    k = 2
    t = 3
    print(solution.containsNearbyAlmostDuplicate(nums,k,t))
