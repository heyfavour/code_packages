class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        return len(s.rstrip().split(" ")[-1])


import torch
from torch import nn

class M(nn.Module):
    def __int__(self):
        pass

    def forward(self,x):
        ouput = nn.Linear(x)
        ouput=ouput+x
        ouput = nn.Linear(ouput)

