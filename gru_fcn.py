import os
import numpy as np
import torch
import torch.nn as nn
from segmentation import calc_segmentation_points
from config import TARGET_SAMPLE_RATE, SUPPORTED_SAMPLE_TYPES
from utils import parse_sample

FILE_PATH = "./models/gru_classification.pth"
FORCE_CPU = True  # 强制使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
N_CLASSES = len(SUPPORTED_SAMPLE_TYPES)  # 分类数
SERIES_TO_ENCODE = ["A", "B", "C"]
CHANNELS = len(SERIES_TO_ENCODE)
TIME_SERIES_DURATION = 20  # 20s
TIME_SERIES_LENGTH = TARGET_SAMPLE_RATE * TIME_SERIES_DURATION  # 采样率*时间，总共的数据点数
POOLING_FACTOR_PER_TIME_SERIES = 5  # 每个时间序列的池化因子,用于降低工作量
SEQ_LENGTH = TIME_SERIES_LENGTH // POOLING_FACTOR_PER_TIME_SERIES  # 降采样后的序列长度


class Vanilla_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Vanilla_GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, seq):
        hidden = torch.zeros(self.num_layers, seq.size(0), self.hidden_size).to(DEVICE)
        # adj_seq = seq.permute(self.batch_size, len(seq), -1)
        output, hidden = self.gru(seq, hidden)
        return output, hidden


class Squeeze_Excite(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, s = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excite(y).view(b, c, 1)
        return x * y.expand_as(x)


class FCN_1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCN_1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=8,
            padding=4,
            padding_mode="replicate",
        )
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm1d(out_channels, eps=1e-03, momentum=0.99)
        self.SE1 = Squeeze_Excite(out_channels)

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels * 2,
            kernel_size=5,
            padding=2,
            padding_mode="replicate",
        )
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm1d(out_channels * 2, eps=1e-03, momentum=0.99)
        self.SE2 = Squeeze_Excite(out_channels * 2)

        self.conv3 = nn.Conv1d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.bn3 = nn.BatchNorm1d(out_channels, eps=1e-03, momentum=0.99)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, seq):
        adj_seq = seq.permute(0, 2, 1)

        y = self.conv1(adj_seq)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.SE1(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.SE2(y)

        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu(y)

        y = self.gap(y)

        return y


class GRU_FCN(nn.Module):
    def __init__(self, seq_len, n_class, dropout_rate, hidden_size):
        super().__init__()
        self.GRU_model = Vanilla_GRU(
            input_size=CHANNELS, hidden_size=hidden_size, num_layers=1
        ).to(DEVICE)
        self.FCN_model = FCN_1D(in_channels=CHANNELS, out_channels=hidden_size).to(
            DEVICE
        )
        self.seq_len = seq_len

        self.dropout = nn.Dropout(p=dropout_rate)
        self.Dense = nn.Linear(in_features=hidden_size * 2, out_features=n_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, seq):
        y_GRU, _ = self.GRU_model(seq)
        y_GRU = y_GRU.transpose(0, 1)[-1]
        y_GRU = self.dropout(y_GRU)
        y_FCN = self.FCN_model(seq).squeeze()
        if len(y_FCN.size()) == 1:
            y_FCN = y_FCN.unsqueeze(0)
        concat = torch.cat([y_GRU, y_FCN], 1)
        y = self.Dense(concat)
        y = self.softmax(y)
        return y


model = GRU_FCN(
    seq_len=SEQ_LENGTH, n_class=N_CLASSES, dropout_rate=0.2, hidden_size=128
).to(DEVICE)


def model_input_parse(sample, segmentations=None, batch_simulation=True):
    """
    将样本转换为模型输入的格式
    """
    if segmentations is None:
        segmentations = calc_segmentation_points(sample)
    sample_array, _ = parse_sample(
        sample,
        segmentations,
        time_series_length=TIME_SERIES_LENGTH,
        pooling_factor_per_time_series=POOLING_FACTOR_PER_TIME_SERIES,
        series_to_encode=SERIES_TO_ENCODE,
    )
    x = sample_array.transpose()

    if batch_simulation:
        x = x[np.newaxis, :, :]

    return x


def predict_raw_input(x):
    assert os.path.exists(FILE_PATH), "model not found，please train first"
    model = torch.load(FILE_PATH, map_location=DEVICE).to(DEVICE)  # 加载模型

    # 转tensor
    x = torch.tensor(x, dtype=torch.float32).to(DEVICE)

    model.eval()  # 验证模式
    with torch.no_grad():
        output = model(x)
    return output


def predict(sample):
    x = model_input_parse(sample, batch_simulation=True)  # 转换为模型输入格式
    output = predict_raw_input(x).squeeze()
    return output
