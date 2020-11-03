class solution:
    @classmethod
    def mysqrt(self, x: int) -> int:
        l,h = 0,x
        while l<=h:
            mid = (l+h)//2
            sqt_x = mid*mid
            if sqt_x == x:
                return mid
            elif sqt_x < x:
                l = mid + 1
            elif sqt_x > x:
                h = mid - 1
        return h


if __name__ == '__main__':
    print(solution.mysqrt(3))
