import collections


class Solution:
    def leastInterval(self, tasks: list[str], n: int) -> int:
        task_count = collections.Counter(tasks)
        tasks_count = sorted(task_count.values(), reverse=True)
        max_count = tasks_count[0]
        col = 0
        for v in tasks_count:
            if v == max_count: col = col + 1
        return max(len(tasks_count), (max_count - 1) * (n + 1) + col)


if __name__ == '__main__':
    tasks = ["A", "A", "A", "B", "B", "B"]
    n = 2
    solution = Solution()
    print(solution.leastInterval(tasks, n))
