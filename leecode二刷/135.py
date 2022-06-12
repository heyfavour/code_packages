class Solution:
    def candy(self, ratings: list[int]) -> int:
        n = len(ratings)
        candy = [1] * n
        for i in range(1, n):  # 后大
            if ratings[i] > ratings[i - 1]: candy[i] = candy[i - 1] + 1
        for i in range(n - 2, -1, -1):  # 前大
            if ratings[i] > ratings[i + 1]: candy[i] = max(candy[i + 1] + 1,candy[i])

        return sum(candy)
if __name__ == '__main__':
    solution = Solution()
    ratings = [1,3,4,5,2]
    print(solution.candy(ratings))
