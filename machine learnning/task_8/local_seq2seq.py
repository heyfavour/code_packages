import torch
from torch import nn
from torch.utils import data
from deal_data import EN2CNDataset
from network import build_model

batch_size = 60

max_output_len = 50  # 最後輸出句子的最大長度
num_steps = 12000  # 總訓練次數
store_steps = 300  # 訓練多少次後須儲存模型
summary_steps = 300  # 訓練多少次後須檢驗是否有overfitting
load_model = False  # 是否需載入模型
store_model_path = "./"  # 儲存模型的位置
load_model_path = None  # 載入模型的位置 e.g. "./ckpt/model_{step}"
data_path = "./data"  # 資料存放的位置

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infinite_iter(data_loader):
    it = iter(data_loader)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(data_loader)


def schedule_sampling():
    return 1


def train(model, optimizer, train_iter, loss_function, total_steps, summary_steps):
    model.train()
    model.zero_grad()
    losses = []
    loss_sum = 0.0
    for step in range(summary_steps):  # 300
        sources, targets = next(train_iter)#[60 50] 英文 [60 50] 中文
        sources, targets = sources.to(device), targets.to(device)
        outputs, preds = model(sources, targets, schedule_sampling())
        # targets 的第一個 token 是 <BOS> 所以忽略
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)
        loss = loss_function(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        loss_sum += loss.item()
        if (step + 1) % 5 == 0:
            loss_sum = loss_sum / 5
            print("\r", "train [{}] loss: {:.3f}, Perplexity: {:.3f}      ".format(total_steps + step + 1, loss_sum,
                                                                                   np.exp(loss_sum)), end=" ")
            losses.append(loss_sum)
            loss_sum = 0.0

    return model, optimizer, losses


def train_process():
    # 準備訓練資料
    train_dataset = EN2CNDataset(data_path, max_output_len, 'training')  # get 50位的英  中
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_iter = infinite_iter(train_loader)
    # 準備檢驗資料
    val_dataset = EN2CNDataset(data_path, max_output_len, 'validation')
    val_loader = data.DataLoader(val_dataset, batch_size=1)
    # 建構模型
    model, optimizer = build_model(train_dataset.en_vocab_size, train_dataset.cn_vocab_size)  # 3922 3805
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    train_losses, val_losses, bleu_scores = [], [], []
    total_steps = 0
    while (total_steps < num_steps):  # 12000
        model, optimizer, loss = train(model, optimizer, train_iter, loss_function, total_steps, summary_steps)  # 0 300
        train_losses += loss
        # 檢驗模型
        val_loss, bleu_score, result = test(model, val_loader, loss_function)
        val_losses.append(val_loss)
        bleu_scores.append(bleu_score)

        total_steps += config.summary_steps
        print("\r", "val [{}] loss: {:.3f}, Perplexity: {:.3f}, blue score: {:.3f}       ".format(total_steps, val_loss,
                                                                                                  np.exp(val_loss),
                                                                                                  bleu_score))

        # 儲存模型和結果
        if total_steps % config.store_steps == 0 or total_steps >= config.num_steps:
            save_model(model, optimizer, config.store_model_path, total_steps)
            with open(f'{config.store_model_path}/output_{total_steps}.txt', 'w') as f:
                for line in result:
                    print(line, file=f)

    return train_losses, val_losses, bleu_scores


if __name__ == '__main__':
    train_losses, val_losses, bleu_scores = train_process()
