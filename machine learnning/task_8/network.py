from torch import nn

emb_dim = 256
hid_dim = 512
n_layers = 3
dropout = 0.5
attention = False  # 是否使用 Attention Mechanism


class Encoder(nn.Module):
    def __init__(self, en_vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        print()
        self.embedding = nn.Embedding(en_vocab_size, emb_dim)#3922 256 输入词汇表 把
        self.hid_dim = hid_dim#512
        self.n_layers = n_layers
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)#256 512 3 0.5
        self.dropout = nn.Dropout(dropout)#0.5

    def forward(self, input):
        embedding = self.embedding(input)#input = [batch size, sequence len, vocab size]
        outputs, hidden = self.rnn(self.dropout(embedding))
        # outputs = [batch size, sequence len, hid dim * directions]
        # hidden =  [num_layers * directions, batch size  , hid dim]
        # outputs 是最上層RNN的輸出

        return outputs, hidden


def build_model(config, en_vocab_size, cn_vocab_size):
    # 建構模型
    encoder = Encoder(en_vocab_size, emb_dim, hid_dim, n_layers, dropout)
    decoder = Decoder(cn_vocab_size, emb_dim, hid_dim, n_layers, dropout, attention)
    model = Seq2Seq(encoder, decoder, device)
    print(model)
    # 建構 optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print(optimizer)
    if config.load_model:
        model = load_model(model, config.load_model_path)
    model = model.to(device)

    return model, optimizer
