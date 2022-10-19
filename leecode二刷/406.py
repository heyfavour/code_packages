class Solution:
    def reconstructQueue(self, people: list[list[int]]) -> list[list[int]]:
        people.sort(key=lambda x: (-x[0], x[1]))
        ans  = []
        for p in people:
            ans.insert(p[1],p)
        return ans



if __name__ == '__main__':
    solution = Solution()
    print(solution.reconstructQueue(people=[[7, 0], [4, 4], [7, 1], [5, 0], [6, 1], [5, 2]]))