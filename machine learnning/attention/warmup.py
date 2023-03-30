"""
https://zhuanlan.zhihu.com/p/461855995
https://www.zhihu.com/question/315772308
"""
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from transformers import optimization
optimizer = None
#300步后学习率变为常数
scheduler = optimization.get_constant_schedule_with_warmup(optimizer,num_warmup_steps=300)
#num_training_steps 代表整个模型训练的step #0-300 warmup 300-1000 decrease
scheduler = optimization.get_linear_schedule_with_warmup(optimizer,num_warmup_steps=300,num_training_steps=1000)
# 多项式衰减学习率
#最后缩小到lr_end
scheduler = optimization.get_polynomial_decay_schedule_with_warmup(optimizer,num_warmup_steps=300,num_training_steps=1000,lr_end=1e-7, power=1.0)
# 余弦循环衰减学习率
scheduler = optimization.get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=300,num_training_steps=1000,num_cycles=1)

