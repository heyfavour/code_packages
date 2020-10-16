from typing import List

class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        m_list = [[0]*n for _ in range(n)]
        directions = [(0,1),(1,0),(0,-1),(-1,0)]
        nums = 0
        def helper(depth):
            nonlocal nums
            if n-(n-depth)*2 <=0 :return
            x = y = n - depth
            for d in directions:
                positions = [(x+d[0]*_,y+d[1]*_) for _ in range(0,n-(n-depth)*2)]
                x,y = positions[-1]
                if len(positions)>=2:
                    positions = positions[:-1]
                    for position in positions:
                        nums = nums + 1
                        m_list[position[0]][position[1]] = nums
                elif len(positions) == 1:
                    position = positions[0]
                    nums = nums + 1
                    m_list[position[0]][position[1]] = nums
                    return
            #结束
            depth = depth - 1
            helper(depth)

        helper(n)
        return m_list
        #print(res)
if __name__ == '__main__':
    s = Solution()
    print(s.generateMatrix(4))
