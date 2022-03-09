class Solution:
    def trap(self, height: list[int]) -> int:
        n = len(height)
        dp = [0] * n

        def get_max_i(max_i, i):
            if height[max_i] <= height[i]: return max_i
            for j in range(i - 1, max_i - 1, -1):
                if height[j] >= height[i]: return j

        def cal_area(left, right):
            area = min((height[left], height[right])) * (right - left - 1) - sum(height[left + 1:right])
            return area

        max_i = 0
        for i in range(n):
            if i == 0:
                dp[0] = 0
            elif (height[i] <= height[i - 1]) or max_i == i - 1:
                dp[i] = dp[i - 1]
            else:
                close_i = get_max_i(max_i, i)
                dp[i] = cal_area(close_i, i) + dp[close_i]
            if height[i] >= height[max_i]: max_i = i
        return dp[-1]


if __name__ == '__main__':
    soution = Solution()
    height = [5,4,1,2]
    print(soution.trap(height))
