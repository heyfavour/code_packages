"""
tensorboard --logdir=log
"""
import torch
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    writer = SummaryWriter("log")
    model = None
    for i in range(200):
        #scalar
        writer.add_scalar('train_acc', 2, global_step=i)
        writer.add_scalar('valid_acc', 4, global_step=i)
        #graph
        init = torch.zeros((1,3,244,244),device="cpu")
        writer.add_graph(model,init)


    writer.close()