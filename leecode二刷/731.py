"""
class MyCalendarTwo:
    def __init__(self):
        self.book1 = []
        self.book2 = []

    def book(self, start: int, end: int) -> bool:
        if any(start<r and end>l for l,r in self.book2):#start end l r  start>=r or end<=l->start<r and end>l
            return False
        for l,r in self.book1:
            if start<r and end>l:
                self.book2.append((max(start,l),min(end,r)))
        self.book1.append((start,end))
        return True
"""
