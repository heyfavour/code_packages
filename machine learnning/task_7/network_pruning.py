import torch
import torch.optim as optim

from torch import nn
from learn_net import StudentNet


def network_slimming(old_model, new_model):
    params = old_model.state_dict()
    new_params = new_model.state_dict()
    selected_idx = []
    for i in range(8):
        conv_num = f'cnn.{i}.1.weight'
        importance = params[conv_num]#BatchNorm2d
        old_dim = len(importance)
        new_dim = len(new_params[conv_num])
        #16 16 32 84 128 256 256 256
        #16 16 32 64 121 243 243 243
        ranking = torch.argsort(importance, descending=True)#对第一维度进行排序
        selected_idx.append(ranking[:new_dim])#BatchNorm2d层

    now_processed = 1
    for name,param in params.items():
        if name.startswith('cnn') and param.size() != torch.Size([]) and now_processed != len(selected_idx):
            if name.startswith(f'cnn.{now_processed}.3'):now_processed=now_processed+ 1#Pointwise的weight 该层处理结束
            if name.endswith('3.weight'):
                if len(selected_idx) == now_processed:
                    new_params[name] = param[:, selected_idx[now_processed - 1]]#7.3weight 原层输出 但是改变顺序
                else:#pointwise
                    #下一次层的顺序 上一层的参数
                    new_params[name] = param[selected_idx[now_processed]][:, selected_idx[now_processed - 1]]
            else:
                new_params[name] = param[selected_idx[now_processed]]#排序后 n.0  conv deepwise n.1 batchnorm n.3 bias
        else:
            new_params[name] = param #n.1 BatchNorm2d的num_batches_tracked 7.3 bias

    new_model.load_state_dict(new_params)
    return new_model


def run_epoch(dataloader, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        # 清空 optimizer
        optimizer.zero_grad()
        # 處理 input
        inputs, labels = batch_data
        inputs = inputs.cuda()
        labels = labels.cuda()

        logits = net(inputs)
        loss = criterion(logits, labels)
        if update:
            loss.backward()
            optimizer.step()

        total_hit += torch.sum(torch.argmax(logits, dim=1) == labels).item()
        total_num += len(inputs)
        total_loss += loss.item() * len(inputs)

    return total_loss / total_num, total_hit / total_num


if __name__ == '__main__':
    net = StudentNet().cuda()
    net.load_state_dict(torch.load('student_custom_small.bin'))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-3)
    now_width_mult = 1
    for i in range(5):
        now_width_mult *= 0.95
        new_net = StudentNet(width_mult=now_width_mult).cuda()
        # [16, 32, 64, 128, 256, 256, 256, 256]
        # [16, 32, 64, 121, 243, 243, 243, 256]
        # [16, 32, 64, 115, 231, 231, 231, 256]
        # [16, 32, 64, 109, 219, 219, 219, 256]
        # [16, 32, 64, 104, 208, 208, 208, 256]
        # [16, 32, 64, 99, 198, 198, 198, 256]
        params = net.state_dict()
        new_net = network_slimming(net, new_net)
