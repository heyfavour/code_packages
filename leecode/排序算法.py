import random
import datetime, time
import copy


def timecal(func):
    def wrapper(*args, **kwargs):
        M = copy.deepcopy(args[1])
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        M.sort()
        if M != result:
            raise Exception(f"未排序:{func.__name__}", M, result)
        print(f"{func.__name__}", t2 - t1)
        return result

    return wrapper


class MySort():
    # 交换排序
    # 冒泡排序
    @timecal
    def maopao(self, sort_list):
        for i in range(len(sort_list) - 1):
            for j in range(len(sort_list) - 1 - i):
                if sort_list[j] > sort_list[j + 1]:
                    sort_list[j], sort_list[j + 1] = sort_list[j + 1], sort_list[j]
        return sort_list

    # 快排
    @timecal
    def kuaipai(self, sort_list):
        def partition(sort_list, start, end):
            pivot = start
            for i in range(start + 1, end + 1):
                if sort_list[i] < sort_list[start]:
                    pivot = pivot + 1
                    sort_list[i], sort_list[pivot] = sort_list[pivot], sort_list[i]
            sort_list[start], sort_list[pivot] = sort_list[pivot], sort_list[start]
            return pivot

        def quick_sort(sort_list, start, end):
            if start < end:
                mid = partition(sort_list, start, end)
                quick_sort(sort_list, start, mid - 1)
                quick_sort(sort_list, mid + 1, end)

        quick_sort(sort_list, 0, len(sort_list) - 1)
        return sort_list

    # 选择排序
    @timecal
    def xuanze(self, sort_list):
        for i in range(len(sort_list) - 1):
            min_index = i
            for j in range(i + 1, len(sort_list)):
                if sort_list[j] < sort_list[min_index]: min_index = j
            sort_list[i], sort_list[min_index] = sort_list[min_index], sort_list[i]
        return sort_list

    @timecal
    def heap_sort(self, sort_list):
        #堆排序
        n = len(sort_list)

        def max_heap(sort_list, n, i):
            largest = i
            l = 2 * i + 1
            r = 2 * i + 2
            if l < n and sort_list[i] < sort_list[l]:
                largest = l
            if r < n and sort_list[largest] < sort_list[r]:
                largest = r
            if largest != i:
                sort_list[i], sort_list[largest] = sort_list[largest], sort_list[i]
                max_heap(sort_list, n, largest)

        for i in range(n, -1, -1):
            max_heap(sort_list, n, i)
        for i in range(n - 1, 0, -1):
            sort_list[i], sort_list[0] = sort_list[0], sort_list[i]
            max_heap(sort_list, i, 0)
        return sort_list

    # 插入排序
    # 简单插入
    @timecal
    def charu(self, sort_list):
        for i in range(1, len(sort_list)):
            sort_value = sort_list[i]
            j = i - 1
            while j >= 0:
                if sort_list[j] > sort_value:
                    sort_list[j + 1] = sort_list[j]
                    j = j - 1
                else:
                    break
            sort_list[j + 1] = sort_value
        return sort_list

    # 希尔排序
    @timecal
    def xier(self, sort_list):
        gap = len(sort_list) // 2
        while gap > 0:
            for i in range(gap, len(sort_list)):
                sort_value = sort_list[i]
                j = i - gap
                while j >= 0:
                    if sort_list[j] > sort_value:
                        sort_list[j + gap] = sort_list[j]
                        j = j - gap
                    else:
                        break
                sort_list[j + gap] = sort_value
            gap = gap // 2
        return sort_list

    @timecal
    def guibing(self, sort_list):
        def megre_sort(sort_list):
            if len(sort_list) < 2: return sort_list
            mid = len(sort_list) // 2
            left = sort_list[:mid]
            right = sort_list[mid:]
            return merge(megre_sort(left), megre_sort(right))

        def merge(left, right):
            result = []
            while left and right:
                if left[0] < right[0]:
                    result.append(left.pop(0))
                else:
                    result.append(right.pop(0))
            result = result + left + right
            return result

        return megre_sort(sort_list)

    @timecal
    def count_sort(self, sort_list):
        # 生产空数组  下标代表了数值大小，值代表个数   用另一种维度储存信息
        bucket = [0] * (max(sort_list) + 1)
        for i in sort_list:
            bucket[i] = bucket[i] + 1
        i = 0
        for j in range(len(bucket)):
            if bucket[j] > 0:
                sort_list[i:i + bucket[j]] = [j] * bucket[j]
                i = i + bucket[j]
        return sort_list

    @timecal
    def bucket_sort(self, sort_list):
        bucket_size = 10
        bucket = [[] for i in range(bucket_size + 1)]
        for i in sort_list:
            bucket[(i - 1) // bucket_size].append(i)
        for i in bucket:
            i.sort()
        index = 0
        for i in bucket:
            for j in i:
                sort_list[index] = j
                index = index + 1
        return sort_list

    @timecal
    def radix_sort(self, sort_list):
        i = 0
        while i < len(str(max(sort_list))):
            bucket = [[] for _ in range(10)]
            for j in sort_list:
                bucket[int(j / (10 ** i)) % 10].append(j)
            sort_list = [x for i in bucket for x in i]
            i = i + 1
        return sort_list


if __name__ == '__main__':
    L = [random.randint(1, 100) for i in range(50000)]

    ms = MySort()
    # 交换排序
    #ms.maopao(copy.deepcopy(L))
    ms.kuaipai(copy.deepcopy(L))
    # 选择排序
    #ms.xuanze(copy.deepcopy(L))
    ms.heap_sort(copy.deepcopy(L))
    # 插入排序
    #ms.charu(copy.deepcopy(L))
    ms.xier(copy.deepcopy(L))
    # 归并排序
    ms.guibing(copy.deepcopy(L))
    # 计数排序 桶排序 基数排序
    ms.count_sort(copy.deepcopy(L))
    ms.bucket_sort(copy.deepcopy(L))
    ms.radix_sort(copy.deepcopy(L))
