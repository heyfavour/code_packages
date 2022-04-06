class Solution:
    # 暴力法
    """
    def nextGreaterElement(self, nums1: list[int], nums2: list[int]) -> list[int]:
        m,n = len(nums1),len(nums2)
        ans = [-1]*m
        for i in range(m):
            j = nums2.index(nums1[i])
            for k in range(j+1,n):
                if nums2[k]>nums1[i]:
                    ans[i] = nums2[k]
                    break
        return ans
    """

    # 单调栈 正向
    def nextGreaterElement(self, nums1: list[int], nums2: list[int]) -> list[int]:
        m, n = len(nums1), len(nums2)
        stack, hash = [], {}
        # num2 找到离他最近得最大得数
        for i in range(n):
            while stack and stack[-1][1] < nums2[i]:
                hash[stack.pop()[1]] = nums2[i]  # 不断将栈里得数据弹出更新
            stack.append((i, nums2[i]))  # i v
        return [hash.get(i, -1) for i in nums1]


if __name__ == '__main__':
    solution = Solution()
    nums1 = [4, 1, 2]
    nums2 = [1, 3, 4, 2]
    print(solution.nextGreaterElement(nums1, nums2))
