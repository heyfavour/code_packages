class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        tmp_dict = {i:i for i in nums if i>0}
        for i in range(len(tmp_dict.keys())):
            i = i + 1
            if tmp_dict.get(i) is None:
                return i

if __name__ == '__main__':
    s = Solution()
    l = [3,4,-1,1]
    l = [0,1,2]
    print(s.firstMissingPositive(l))
