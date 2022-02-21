class Solution:
    def pushDominoes(self, dominoes: str) -> str:
        dominoes = list(dominoes)
        dirs = []
        n = len(dominoes)
        for i, v in enumerate(dominoes):
            if v == ".": continue
            dirs.append((i, v))
        if dirs == []:return "".join(dominoes)
        if dirs[0][1] == "L": dominoes[0:dirs[0][0]] = "L" * dirs[0][0]
        if dirs[-1][1] == "R": dominoes[dirs[-1][0] + 1:] = "R" * (n - dirs[-1][0] - 1)
        for i in range(len(dirs) - 1):
            left, left_dir = dirs[i][0], dirs[i][1]
            right, right_dir = dirs[i + 1][0], dirs[i + 1][1]
            left_right_len = right - left - 1
            left_right_mid = left_right_len // 2
            if left_dir == right_dir:
                dominoes[left + 1:right] = left_right_len * left_dir
            elif left_dir == "L" and right_dir == "R":
                    continue
            elif left_dir == "R" and right_dir == "L":
                dominoes[left + 1:left + left_right_mid + 1] = left_dir * left_right_mid
                dominoes[right - left_right_mid: right] = right_dir * left_right_mid
        return "".join(dominoes)


if __name__ == '__main__':
    solution = Solution()
    dominoes = "RRLL..RRLLRRRRR....."
    # "LL.RR.LLRRLL.."
    print(solution.pushDominoes(dominoes))
