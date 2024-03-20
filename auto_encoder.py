"""
降噪自编码器 Denoising Auto-Encoder
采用正常时间序列无监督训练，用于产生是否异常的置信度
该置信度会用于之后的分类，以降低假阳率
"""
import os
import numpy as np
import torch
import torch.nn as nn
from utils import parse_sample, save_args,load_model
from config import FILE_OUTPUT, FORCE_CPU, TARGET_SAMPLE_RATE, SUPPORTED_SAMPLE_TYPES

POOLING_FACTOR_PER_TIME_SERIES = 5  # 每条时间序列的降采样因子
TIME_SERIES_DURATION = 10  # 输入模型的时间序列时长为10s
TIME_SERIES_LENGTH = TARGET_SAMPLE_RATE * TIME_SERIES_DURATION  # 时间序列长度
SERIES_TO_ENCODE = ["A", "B", "C"]  # 参与训练和预测的序列，power暂时不用
CHANNELS = len(SERIES_TO_ENCODE)
TOTAL_LENGTH = TIME_SERIES_LENGTH // POOLING_FACTOR_PER_TIME_SERIES
TOTAL_LENGTH *= CHANNELS  # 输入总长度

FILE_PATH = "./models/auto_encoder/"  # 模型保存路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
# print("Using device:", DEVICE)


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


loss_func = nn.MSELoss()  # 损失函数
models = {} # 预载入的所有模型
for type in SUPPORTED_SAMPLE_TYPES:
    model_path = f"{FILE_PATH}{type}.pth"
    model = BP_AE(seq_len=TOTAL_LENGTH, latent_dim=round(TOTAL_LENGTH / 5)).to(DEVICE)
    model.load_state_dict(load_model(model_path, DEVICE))
    models[type] = model


def predict_raw_input(x):
    assert x.dim() == 1  # 一维
    assert len(x) == TOTAL_LENGTH  # 确保长度正确
    results = {}
    losses = {}
    for type in SUPPORTED_SAMPLE_TYPES:
        model = models[type]
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


def visualize_prediction_result(uuid, y_before, results, losses):
    y_before = y_before.view(CHANNELS, -1).cpu().numpy()
    for i, ae_type in enumerate(SUPPORTED_SAMPLE_TYPES):
        loss = losses[i]
        y_after = results[ae_type]
        y_after = y_after.view(CHANNELS, -1).cpu().numpy()

        kwargs = {
            "channels": CHANNELS,
            "series_to_encode": SERIES_TO_ENCODE,
            "ae_type": ae_type,
            "loss": loss,
            "y_before": y_before,
            "y_after": y_after,
        }

        save_args(f"./file_output/{uuid}/AE/raw/{ae_type}.pkl", kwargs)


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


def predict(uuid, sample):
    x = model_input_parse(sample)
    results, losses, confidences = predict_raw_input(x)
    if FILE_OUTPUT:
        visualize_prediction_result(uuid, x, results, losses)
    return torch.tensor(list(confidences.values()), dtype=torch.float).to(DEVICE)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
