"""
降噪自编码器 Denoising Auto-Encoder
采用正常时间序列无监督训练，用于产生是否异常的置信度
该置信度会用于之后的分类，以降低假阳率
"""
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
from utils import parse_sample
from config import DEBUG, TARGET_SAMPLE_RATE, SUPPORTED_SAMPLE_TYPES

POOLING_FACTOR_PER_TIME_SERIES = 5  # 每条时间序列的采样点数
TIME_SERIES_DURATION = 10  # 输入模型的时间序列时长为10s
TIME_SERIES_LENGTH = TARGET_SAMPLE_RATE * TIME_SERIES_DURATION  # 时间序列长度
SERIES_TO_ENCODE = ["A", "B", "C"]  # 参与训练和预测的序列，power暂时不用
CHANNELS = len(SERIES_TO_ENCODE)
TOTAL_LENGTH = TIME_SERIES_LENGTH // POOLING_FACTOR_PER_TIME_SERIES
TOTAL_LENGTH *= CHANNELS  # 输入总长度

FILE_PATH = "./models/auto_encoder/"  # 模型保存路径
FORCE_CPU = True  # 强制使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
print("Using device:", DEVICE)


class BP_AE(nn.Module):
    def __init__(self, seq_len, latent_dim):
        super(BP_AE, self).__init__()
        self.bottle_neck_output = None

        self.encoder = nn.Sequential(
            nn.Linear(seq_len, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, seq_len),
        )

    def forward(self, x):
        x = self.encoder(x)
        self.bottle_neck_output = x
        x = self.decoder(x)
        return x


model = BP_AE(seq_len=TOTAL_LENGTH, latent_dim=round(TOTAL_LENGTH / 5)).to(DEVICE)
print(model)

loss_func = nn.MSELoss()  # 损失函数


def predict_raw_input(x):
    assert x.dim() == 1  # 一维
    assert len(x) == TOTAL_LENGTH  # 确保长度正确
    results = {}
    losses = {}
    for type in SUPPORTED_SAMPLE_TYPES:
        model_path = f"{FILE_PATH}{type}.pth"
        assert os.path.exists(model_path), f"model {type} not found, please train first"
        model = torch.load(model_path, map_location=DEVICE).to(DEVICE)
        model.eval()
        with torch.no_grad():
            result = model(x)
            results[type] = result
            loss = loss_func(result, x)
            losses[type] = loss.item()
    losses = list(losses.values())
    # 使用sigmoid函数将loss转换为概率
    confidences = [-loss * 100 for loss in losses]
    # 和为1
    confidences = softmax(confidences)
    confidences = [round(confidence, 2) for confidence in confidences]
    # key还原上
    confidences = dict(zip(SUPPORTED_SAMPLE_TYPES, confidences))
    return results, losses, confidences


def visualize_prediction_result(y_before, results, losses):
    for i, ae_type in enumerate(SUPPORTED_SAMPLE_TYPES):
        loss = losses[i]
        y_after = results[ae_type]
        draw(
            y_before,
            y_after,
            filename=f"./debug_output/AE/{ae_type}",
            title=f"AutoEncoder type: {ae_type} - loss: {loss}",
        )


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
    result = result.reshape(TOTAL_LENGTH)
    return torch.tensor(result, dtype=torch.float).to(DEVICE)


def draw(y_before, y_after, filename, title=""):
    y_before = y_before.view(CHANNELS, -1)
    y_after = y_after.view(CHANNELS, -1)
    figure, (axes) = plt.subplots(CHANNELS, 1, figsize=(12, 5), dpi=150)
    for i in range(CHANNELS):
        ax = axes[i]
        ax.plot(y_before[i], label="original")
        ax.plot(y_after[i], label="AutoEncoder result")
        ax.set_title(f"Channel: {SERIES_TO_ENCODE[i]}")
        ax.set_xlim(0, None)
        ax.set_ylim(bottom=0, top=5)

    figure.suptitle(title)
    lines, labels = figure.axes[-1].get_legend_handles_labels()
    figure.legend(lines, labels, loc="upper right")
    figure.set_tight_layout(True)
    plt.savefig(filename)
    plt.close()


def predict(sample):
    x = model_input_parse(sample)
    results, losses, confidences = predict_raw_input(x)
    if DEBUG:
        visualize_prediction_result(x, results, losses)
    return torch.tensor(list(confidences.values()), dtype=torch.float).to(DEVICE)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def sigmoid_d(x):
    return np.exp(-x) / (1 + np.exp(-x)) ** 2
