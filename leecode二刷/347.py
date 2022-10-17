import heapq


class Solution:
    def topKFrequent(self, nums: list[int], k: int) -> list[int]:
        topkdict = {}
        for i in nums: topkdict[i] = topkdict.get(i, 0) + 1
        topklist = []
        for num, count in topkdict.items(): heapq.heappush(topklist, (-count, num))
        ans = []
        for i in range(k): ans.append(heapq.heappop(topklist)[1])
        return ans


if __name__ == '__main__':
    solutin = Solution()
    print(solutin.topKFrequent(nums=[1, 1, 1, 2, 2, 3], k=2))
