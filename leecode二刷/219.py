import setuptools.config


class Solution:
    def containsNearbyDuplicate(self, nums: list[int], k: int) -> bool:
        # 慢
        # _count = collections.defaultdict(list)
        # for i in range(len(nums)):
        #     _count[nums[i]].append(i)
        #     v = _count[nums[i]]
        #     if len(v)>=2 and (v[-1]-v[-2]) <=k:return True
        # return False
        # 滑动窗口 List 慢 利用queue无重复特性 这里使用set
        # from collections import deque
        # queue = deque()
        # for i in range(len(nums)):
        #     queue.append(nums[i])
        #     if i>=k:queue.popleft()
        #     if nums[i] in queue:return True
        # return False
        # 0 1 2 3
        s = set()
        for i in range(len(nums)):
            if nums[i] in s:return True
            s.add(nums[i])
            if i>=k:s.remove(nums[i-k])
        return False