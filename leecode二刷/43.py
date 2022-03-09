class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        def multiply_help(num,single_num):
            _add = 0
            ans = ""
            for i in num[::-1]:
                _m = int(i)*single_num + _add
                _add,_num = _m//10,str(_m%10)
                ans = _num + ans
            if _add:ans = str(_add) + ans
            return ans

        def sum_help(num1,num2):
            print(num1,num2)
            num1=list(num1)
            num2=list(num2)
            _add = 0
            n1 = len(num1)
            n2 = len(num2)
            n = max((n1,n2))
            string = ""
            while num1 or num2 or _add:
                val_1,val_2 = 0,0
                if num1:
                    val_1 = int(num1[-1])
                    num1.pop()
                if num2:
                    val_2 = int(num2[-1])
                    num2.pop()

                _sum = int(val_1)+int(val_2) + _add
                _add,str_num = _sum//10,_sum%10
                string = str(str_num) + string
            return string
        sum_list = []
        for i,v in enumerate(num2[::-1]):
            sum_list.append(multiply_help(num1,int(v))+"0"*i)
        print(sum_list)
        all = "0"
        for i in sum_list:
            all = sum_help(all,i)
        return all





if __name__ == '__main__':
    num1 = "123"
    num2 = "456"
    solution = Solution()
    print(solution.multiply(num1,num2))