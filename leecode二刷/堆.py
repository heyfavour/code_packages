import heapq
#由于实现优先队列
stack = []
#最小堆
heapq.heapify( stack )#线性时间内将list转成堆
heapq.heappush(stack,(1,object))
heapq.heappop(stack)
heapq.heapreplace(stack,(1,object))
heapq.nlargest()
heapq.nsmallest()
#最大堆
heapq._heapify_max(stack)
heapq._heappop_max(stack)
heapq._heapreplace_max(stack,(1,object))



from queue import PriorityQueue
#PriorityQueue
#本质也是heapq