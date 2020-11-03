class Solution:
    @classmethod
    def addBinary(self, a: str, b: str) -> str:
        a, b = [int(i) for i in a], [int(i) for i in b]
        a.reverse()
        b.reverse()
        sum_list = []
        add = 0
        for i in range(max(len(a), len(b))):
            sum = (a[i] if i<=len(a)-1 else 0)   + (b[i] if i<=len(b)-1 else 0) + add
            if sum >= 2:
                add = 1
                sum = sum%2
            else:
                add = 0
            sum_list.append(sum)
            if i == max(len(a), len(b)) - 1 and add > 0:
                sum_list.append(add)
        sum_list.reverse()
        return "".join([str(i) for i in sum_list])


if __name__ == '__main__':
    a = "11"
    b = "1"
    print(Solution.addBinary(a, b))
