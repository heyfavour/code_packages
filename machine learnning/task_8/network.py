import torch
from torch import nn

emb_dim = 256
hid_dim = 512
n_layers = 3
dropout = 0.5
attention = False  # 是否使用 Attention Mechanism
learning_rate = 0.00005


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.hid_dim = hid_dim

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs = [batch size, sequence len, hid dim * directions]
        # decoder_hidden = [num_layers, batch size, hid dim]
        # 一般來說是取 Encoder 最後一層的 hidden state 來做 attention
        ########
        # TODO #
        ########
        attention = None

        return attention


class Encoder(nn.Module):
    def __init__(self, en_vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(en_vocab_size, emb_dim)  # 3922 256 输入词汇表 把
        self.hid_dim = hid_dim  # 512
        self.n_layers = n_layers
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True,
                          bidirectional=True)  # 256 512 3 0.5
        self.dropout = nn.Dropout(dropout)  # 0.5

    def forward(self, input):  # [60 50]
        print(input.size())
        embedding = self.embedding(input)  # [60 50] => [60 50 256]
        outputs, hidden = self.rnn(self.dropout(embedding))  # [60 50 1024]
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, cn_vocab_size, emb_dim, hid_dim, n_layers, dropout, isatt):
        super().__init__()
        self.cn_vocab_size = cn_vocab_size
        self.hid_dim = hid_dim * 2
        self.n_layers = n_layers
        self.embedding = nn.Embedding(cn_vocab_size, emb_dim)  # 3805 256
        self.dropout = nn.Dropout(dropout)
        self.isatt = isatt  # False
        self.attention = Attention(hid_dim)
        # 如果使用 Attention Mechanism 會使得輸入維度變化，請在這裡修改
        # e.g. Attention 接在輸入後面會使得維度變化，所以輸入維度改為
        # self.input_dim = emb_dim + hid_dim * 2 if isatt else emb_dim
        self.input_dim = emb_dim  # 256
        self.rnn = nn.GRU(self.input_dim, self.hid_dim, self.n_layers, dropout=dropout,
                          batch_first=True)  # 256 512 3 0.5
        self.embedding2vocab1 = nn.Linear(self.hid_dim, self.hid_dim * 2)
        self.embedding2vocab2 = nn.Linear(self.hid_dim * 2, self.hid_dim * 4)
        self.embedding2vocab3 = nn.Linear(self.hid_dim * 4, self.cn_vocab_size)

    def forward(self, input, hidden, encoder_outputs):#[1]*60 [3 60 1024] [6 50 1024]
        # Decoder 只會是單向，所以 directions=1
        input = input.unsqueeze(1)  # [60 1]
        embedded = self.dropout(self.embedding(input))  # [60 1 256] [batch size, 1, emb dim]
        if self.isatt:
            attn = self.attention(encoder_outputs, hidden)  # hidden=[batch size, n layers * directions, hid dim]
            # TODO: 在這裡決定如何使用 Attention，e.g. 相加 或是 接在後面， 請注意維度變化
        output, hidden = self.rnn(embedded, hidden)  # 60 1 256
        # 將 RNN 的輸出轉為每個詞出現的機率
        output = self.embedding2vocab1(output.squeeze(1))
        output = self.embedding2vocab2(output)
        prediction = self.embedding2vocab3(output)  # 60  3805 [batch size, vocab size]
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input, target, teacher_forcing_ratio):
        # input  = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        # teacher_forcing_ratio 是有多少機率使用正確答案來訓練
        batch_size = target.shape[0]  # 60
        target_len = target.shape[1]  # 50
        vocab_size = self.decoder.cn_vocab_size  # 3805
        encoder_outputs, hidden = self.encoder(input)  # [60 50 512*2] [6 60 512]
        # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)  # 3 2 60 -1 [3 2 60 512]
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)  # 3 60 1024
        input = target[:, 0]  ## 取的 <BOS> token [1]*60
        preds = []

        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)  # 準備一個儲存空間來儲存輸出 [60 50 3805]

        for t in range(1, target_len):#50
            print(t)
            output, hidden = self.decoder(input, hidden, encoder_outputs)# [1]*60 [3 60 1024] [6 50 1024]
            outputs[:, t] = output
            # 決定是否用正確答案來做訓練
            teacher_force = random.random() <= teacher_forcing_ratio
            # 取出機率最大的單詞
            top1 = output.argmax(1)
            # 如果是 teacher force 則用正解訓練，反之用自己預測的單詞做預測
            input = target[:, t] if teacher_force and t < target_len else top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds

    def inference(self, input, target):
        ########
        # TODO #
        ########
        # 在這裡實施 Beam Search
        # 此函式的 batch size = 1
        # input  = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        batch_size = input.shape[0]
        input_len = input.shape[1]  # 取得最大字數
        vocab_size = self.decoder.cn_vocab_size

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, input_len, vocab_size).to(self.device)
        # 將輸入放入 Encoder
        encoder_outputs, hidden = self.encoder(input)  # [60 50 512*2]
        print(hidden.size())
        # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        # 取的 <BOS> token
        input = target[:, 0]
        preds = []
        for t in range(1, input_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            # 將預測結果存起來
            outputs[:, t] = output
            # 取出機率最大的單詞
            top1 = output.argmax(1)
            input = top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds


def build_model(en_vocab_size, cn_vocab_size):  # 3922 3805
    # 建構模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(en_vocab_size, emb_dim, hid_dim, n_layers, dropout)  # 3922 256 512 3 0.5
    decoder = Decoder(cn_vocab_size, emb_dim, hid_dim, n_layers, dropout, attention)  # 3805 256 512 3 0.5
    model = Seq2Seq(encoder, decoder, device)
    # 建構 optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    return model, optimizer


if __name__ == '__main__':
    build_model(3922, 3805)
