import torch
import numpy as np
from torch import nn
from torch.utils import data
from deal_data import EN2CNDataset
from network import build_model

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

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
    model.train()  # Seq2Seq
    model.zero_grad()
    losses = []
    loss_sum = 0.0
    for step in range(summary_steps):  # 300
        sources, targets = next(train_iter)  # [60 50] 英文 [60 50] 中文
        sources, targets = sources.to(device), targets.to(device)
        outputs, preds = model(sources, targets, schedule_sampling())  # Seq2Seq [60 50 3805] [60 49]
        # targets 的第一個 token 是 <BOS> 所以忽略
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)
        loss = loss_function(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # 梯度裁剪
        optimizer.step()

        loss_sum += loss.item()
        if (step + 1) % 5 == 0:
            loss_sum = loss_sum / 5
            print(f"train [{total_steps + step + 1}] loss: {loss_sum:.3f}, Perplexity: {np.exp(loss_sum):.3f}")
            losses.append(loss_sum)
            loss_sum = 0.0

    return model, optimizer, losses


def tokens2sentence(outputs, int2word):  # 数字转句子
    sentences = []
    for tokens in outputs:
        sentence = []
        for token in tokens:
            word = int2word[str(int(token))]
            if word == '<EOS>':
                break
            sentence.append(word)
        sentences.append(sentence)

    return sentences


def computebleu(sentences, targets):  # 计算得分
    score = 0

    def cut_token(sentence):
        tmp = []
        for token in sentence:
            if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
                tmp.append(token)
            else:
                tmp += [word for word in token]
        return tmp

    for sentence, target in zip(sentences, targets):
        sentence = cut_token(sentence)
        target = cut_token(target)
        score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0))

    return score


def test(model, dataloader, loss_function):
    model.eval()
    loss_sum, bleu_score = 0.0, 0.0
    n = 0
    result = []
    for sources, targets in dataloader:
        sources, targets = sources.to(device), targets.to(device)
        batch_size = sources.size(0)
        outputs, preds = model.inference(sources, targets)
        # targets 的第一個 token 是 <BOS> 所以忽略
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)

        loss = loss_function(outputs, targets)
        loss_sum += loss.item()

        # 將預測結果轉為文字
        targets = targets.view(sources.size(0), -1)
        preds = tokens2sentence(preds, dataloader.dataset.int2word_cn)  # 数字转句子
        sources = tokens2sentence(sources, dataloader.dataset.int2word_en)  # 数字转句子
        targets = tokens2sentence(targets, dataloader.dataset.int2word_cn)  # 数字转句子
        for source, pred, target in zip(sources, preds, targets):
            result.append((source, pred, target))
        # 計算 Bleu Score
        bleu_score = bleu_score + computebleu(preds, targets)  # 计算得分

        n = n + batch_size

    return loss_sum / len(dataloader), bleu_score / n, result


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

        total_steps = total_steps + summary_steps
        print(
            "val [{total_steps}] loss: {val_loss:.3f}, Perplexity: {np.exp(val_loss):.3f}, blue score: {bleu_score:.3f}")

    return train_losses, val_losses, bleu_scores


if __name__ == '__main__':
    train_losses, val_losses, bleu_scores = train_process()
