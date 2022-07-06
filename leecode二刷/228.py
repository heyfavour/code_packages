class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        vote_1,count_1 = None,0
        vote_2,count_2 = None,0

        for num in nums:
            if vote_1 == num:count_1 = count_1+1
            elif vote_2 == num:count_2 = count_2+1
            elif count_1 == 0:vote_1,count_1 = num,1
            elif count_2 == 0:vote_2,count_2 = num,1
            else:
                count_1 = count_1 - 1
                count_2  = count_2 - 1
        count_1,count_2 =0,0
        for num in nums:
            if num == vote_1:count_1=count_1+1
            if num == vote_2:count_2=count_2+1
        ans = []

        if count_1>len(nums)/3:ans.append(vote_1)
        if count_2>len(nums)/3:ans.append(vote_2)
        return ans


