class Solution:
    def fullJustify(self, words: list[str], maxWidth: int) -> list[str]:
        ans = []
        iword = 0
        while iword <= len(words) - 1:
            row, row_len = [], 0
            while iword <= len(words) - 1:
                next_len = row_len + (1 if row else 0) + len(words[iword])
                if next_len > maxWidth:break
                row_len = next_len
                row.append(words[iword])
                iword = iword + 1
            if len(row)>1:
                space_num = (maxWidth - row_len)
                print(space_num)
                every, add = space_num // (len(row) - 1), space_num % (len(row) - 1)
                row_string = row[0]
                for i in range(1, len(row)):
                    row_string = row_string + (1 + every + (1 if i <= add else 0)) * " " + row[i]
            else:
                row_string = row[0]+(maxWidth-len(row[0]))*" "
            ans.append(row_string)
        return ans

if __name__ == '__main__':
    words = ["This", "is", "an", "example", "of", "text", "justification."]
    maxWidth = 16
    solution = Solution()
    print(solution.fullJustify(words,maxWidth))
