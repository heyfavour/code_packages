class solution:
    @classmethod
    def islongpressedname(self, name: str, typed: str) -> bool:
        # 双指针
        i, j = 0, 0
        while j < len(typed):
            if i < len(name) and name[i] == typed[j]:
                i = i + 1
                j = j + 1
            elif j > 0 and typed[j] == typed[j - 1]:
                j = j + 1
            else:
                return False
        return i == len(name)


if __name__ == '__main__':
    name = "alex1"
    typed = "aaleex"
    print(solution.islongpressedname(name, typed))
