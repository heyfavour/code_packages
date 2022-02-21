int_max = 2 ** 32
int_min = -2 ** 32
"""
         ""      +/1   number   other  
start   start   sign   number    end
sign    end     end    number    end
number  end     end    number    end    
end     end     end    end       end 
"""


class Automaton():
    def __init__(self):
        self.state = "start"
        self.sign = 1
        self.ans = 0
        self.table = {
            "start": ["start", "sign", "number", "end"],
            "sign": ["end", "end", "number", "end"],
            "number": ["end", "end", "number", "end"],
            "end": ["end", "end", "end", "end"],
        }

        def get_col(col_str):
            if col_str.isspace():return 0
            elif col_str == "+" or col_str == "-":return 1
            elif col_str
