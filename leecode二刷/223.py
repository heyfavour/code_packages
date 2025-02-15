class Solution:
    def computeArea(self, ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int) -> int:
        S1 = (ax2-ax1)*(ay2-ay1)
        S2 = (bx2-bx1)*(by2-by1)

        W = max(min(ax2,bx2)-max(bx1,ax1),0)
        H = max(min(ay2,by2)-max(ay1,by1),0)
        S = W*H
        return S1+S2-S