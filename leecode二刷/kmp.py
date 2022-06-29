def build_next(patt):
    #计算next数组
    next = [0]#初始数组
    prefix_len = 0#当前最长公共前后缀长度
    i = 1
    while i<len(patt):
        if patt[i] == patt[prefix_len]:
            prefix_len = prefix_len+1
            i = i+1
            next.append(prefix_len)
        else:
            if prefix_len == 0:
                i = i + 1
                next.append(0)
            else:
                prefix_len = next[prefix_len-1]
    return next

def kmp_search(string,patt):
    next = build_next(patt)
    i,j = 0,0
    while i<len(string):
        if string[i] == patt[j]:
            i = i+1
            j = j+1
        elif j>0:
            j = next[j-1]
        else:
            i=i+1
        if j == len(patt):return i-j

if __name__ == '__main__':
    string = "ABCABABCAA"
    patt = "ABABC"
    print(kmp_search(string,patt))

