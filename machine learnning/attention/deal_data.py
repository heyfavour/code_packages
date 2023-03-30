import os
import json
import time

import torch
import random
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

class SpeakerDataset(Dataset):
    def __init__(self, segment_len=128):
        self.segment_len = segment_len
        # Load the mapping from speaker neme to their corresponding id.
        mapping_path = Path("./data") / "mapping.json"  # pathlib.WindowsPath
        #mapping = {id2speaker:{id:speaker},speaker2id:{speaker:id}}
        mapping = json.load(mapping_path.open())
        self.speaker2id = mapping["speaker2id"]

        # Load metadata of training data.
        metadata_path = "./data/metadata.json"
        import pprint
        metadata = json.load(open(metadata_path))["speakers"]#["n_mels","speakers"] {speaker:[{feature_path:'xxx.pt',len:xxx}]}
        # Get the total number of speaker.
        self.speaker_num = len(metadata.keys())#600
        self.data = []
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"], self.speaker2id[speaker]])#[uttr-xxxx,id]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        # Load preprocessed mel-spectrogram.
        mel = torch.load(os.path.join("./data", feat_path))

        # Segmemt mel-spectrogram into "segment_len" frames.
        if len(mel) > self.segment_len:
            start = random.randint(0, len(mel) - self.segment_len)
            mel = torch.FloatTensor(mel[start:start + self.segment_len])
        else:
            mel = torch.FloatTensor(mel)
        speaker = torch.LongTensor([speaker])
        return mel, speaker#[128,40] [1] id


def collate_batch(batch):#[[128 40] id]*32
    """
    torch.nn.utils.rnn.pad_sequence(sequences, batch_first=False, padding_value=0.0)
    参数说明：sequences：输入的tensor数据，类型为列表或者tuple等等
            batch_first：决定输出中batch这一维度是否在第一维
            padding_value：要填充的值，一般为0
    """
    mel, speaker = zip(*batch) # mel: (batch size, 128, 40)
    pad_mel = pad_sequence(mel, batch_first=True, padding_value=-20)
    return pad_mel, torch.LongTensor(speaker)

def get_dataloader():
    """Generate dataloader"""
    dataset = SpeakerDataset()#56666
    speaker_num = dataset.speaker_num
    # Split dataset into  dataset and validation dataset
    trainlen = int(0.9 * len(dataset))#59999*0.9=50999
    lengths = [trainlen, len(dataset) - trainlen]#[50999,5667][0.9 0.1]
    trainset, validset = random_split(dataset, lengths)#[50999 5667] 切片做子集
    """
    drop_last：告诉如何处理数据集长度除于batch_size余下的数据。True就抛弃，否则保留 
    pin_memory: 如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存
    collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能。
    """
    train_loader = DataLoader(trainset,batch_size=64,shuffle=True,drop_last=True,num_workers=0,pin_memory=True,collate_fn=collate_batch)
    valid_loader = DataLoader(validset,batch_size=32,num_workers=0,drop_last=True,pin_memory=True,collate_fn=collate_batch)

    return train_loader, valid_loader, speaker_num


if __name__ == '__main__':
    # speaker = SpeakerDataset()
    train_loader, valid_loader, speaker_num = get_dataloader()

