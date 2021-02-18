n=1
count, i = 0, 0
n = str(n)
base = 1


def str2int(num):
    if not num: return 0
    return int(num)


for i in range(len(n)):
    pre, k, aft = str2int(n[:len(n) - i - 1]), str2int(n[len(n) - i - 1]), str2int(n[len(n) - i:])
    print(pre,k,aft)
    if k >= 1: count = count + (pre + 1) * base
    if k < 1: count = count + pre * base
    if k == 1: count = count + pre * base + aft + 1
    base = base*10
print(count)

