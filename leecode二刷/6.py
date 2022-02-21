class Solution:
    def convert(self, s: str, numRows: int) -> str: #4
        if len(s) <= numRows:return s
        if numRows==1:return s
        l = [[]*numRows for _ in range(numRows)]
        for k,v in enumerate(s):
            row = (k+1) % (2*numRows-2)
            if 0<row <=numRows:
                l[row-1].append(v)
            elif row==0:
                l[1].append(v)
            else:
                l[-1*(row-numRows) -1].append(v)
        return "".join(["".join(i) for i in l])

if __name__ == '__main__':
    solution = Solution()
    s = "PAYPALISHIRING"
    numRows = 4
    print(solution.convert(s,numRows ))