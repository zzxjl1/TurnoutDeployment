"""
训练用GRU对时间序列上的每一点作为分割点的可能性进行预测
这个模型的输出会用于和原文中的分割算法结合，提升分割的准确性
RNN对序列敏感，因此能够捕捉到stage切换间的变化
"""
import os
import numpy as np
import torch
import torch.nn as nn
from config import TARGET_SAMPLE_RATE, FORCE_CPU
from utils import find_nearest, parse_sample, load_model

FILE_PATH = "./models/gru_score.pth"
TIME_SERIES_DURATION = 15  # 15s
TIME_SERIES_LENGTH = TARGET_SAMPLE_RATE * TIME_SERIES_DURATION  # 采样率*时间，总共的数据点数
SERIES_TO_ENCODE = ["A", "B", "C"]  # 生成三相电流序列，不生成power曲线
POOLING_FACTOR_PER_TIME_SERIES = 3  # 每个时间序列的池化因子,用于降低工作量
SEQ_LENGTH = TIME_SERIES_LENGTH // POOLING_FACTOR_PER_TIME_SERIES  # 降采样后的序列长度


DEVICE = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
CHANNELS = len(SERIES_TO_ENCODE)  # 通道数


class GRUScore(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.5):
        super(GRUScore, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=self.dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Linear(64, 8),
            nn.Linear(8, 1),
        )
        self.activation = nn.ReLU()

        self.init_weights()

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=DEVICE)
        # Forward propagate RNN
        out, _ = self.gru(x, h0)
        # Decode the hidden state of the last time step
        out = self.fc(out)
        out = self.activation(out)
        return out

    # 权重初始化
    def init_weights(self):
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                param.data[self.hidden_size : 2 * self.hidden_size] = 1
        for name, param in self.fc.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
        for name, param in self.activation.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)


model = GRUScore(input_size=CHANNELS, hidden_size=SEQ_LENGTH, output_size=1).to(DEVICE)
model.load_state_dict(load_model(FILE_PATH, DEVICE))


def predict(t) -> np.ndarray:
    batch_size, channels, seq_len = t.shape
    assert batch_size == 1
    assert channels == CHANNELS
    assert seq_len == SEQ_LENGTH
    input = t.transpose(0, 2, 1)
    input = torch.from_numpy(input).float().to(DEVICE)
    out = model(input)
    out = out.detach().cpu().numpy()
    return out.squeeze()


def get_score_by_time(out, sec):
    x, y = model_output_to_xy(out)
    index = find_nearest(x, sec)
    return y[index]


def time_to_index(sec):
    return int(sec * TARGET_SAMPLE_RATE // POOLING_FACTOR_PER_TIME_SERIES)


def model_output_to_xy(out, end_sec=None):
    x = np.arange(0, TIME_SERIES_DURATION, TIME_SERIES_DURATION / len(out))
    y = out
    if end_sec:
        index = time_to_index(end_sec)
        x = x[:index]
        y = y[:index]
    return x, y


def model_input_parse(sample):
    """
    将样本转换为模型输入的格式
    """
    result, _ = parse_sample(
        sample,
        segmentations=None,
        time_series_length=TIME_SERIES_LENGTH,
        pooling_factor_per_time_series=POOLING_FACTOR_PER_TIME_SERIES,
        series_to_encode=SERIES_TO_ENCODE,
    )
    result = result.reshape(1, CHANNELS, SEQ_LENGTH)
    return result
