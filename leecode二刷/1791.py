class Solution:
    """
    def findCenter(self, edges: list[list[int]]) -> int:
        return  edges[0][0] if edges[0][0] in edges[1] else edges[0][1]
    """
    def findCenter(self, edges: list[list[int]]) -> int:
        du_dict = {}
        for i in edges:
            du_dict[i[0]] = du_dict.get(i[0],0)+ 1
            du_dict[i[1]] = du_dict.get(i[1],0)+ 1
        print(du_dict)
        for k,v in du_dict.items():
            if v == len(edges) :return k

if __name__ == '__main__':
    edges = [[1, 2], [2, 3], [4, 2]]
    solution = Solution()
    print(solution.findCenter(edges))

