class Solution:
    def simplifyPath(self, path: str) -> str:
        path = path.split("/")
        path_list = []
        for i in path:
            if i == "":
                continue
            elif i == "..":
                if path_list: path_list.pop()
            elif i == ".":
                continue
            else:
                path_list.append(i)

        return "/" + '/'.join(path_list)


if __name__ == '__main__':
    path = "/home//foo/../"
    solution = Solution()
    print(solution.simplifyPath(path))
