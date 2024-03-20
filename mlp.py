"""
pytorch implementation of dnn classification
用dnn对提取的特征进行分类,代替原文中的svm
"""

import os
import torch
import torch.nn as nn
from extract_features import IGNORE_LIST, calc_features, SERIES_TO_ENCODE
from utils import load_model
from config import SUPPORTED_SAMPLE_TYPES, FILE_OUTPUT, FORCE_CPU

FILE_PATH = "./models/mlp_classification.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
# print("Using device:", DEVICE)

INPUT_VECTOR_SIZE = 1 + 15 * len(SERIES_TO_ENCODE) * 3 - len(IGNORE_LIST)  # 输入向量的大小
N_CLASSES = len(SUPPORTED_SAMPLE_TYPES)  # 分类数


MLP = nn.Sequential(
    nn.BatchNorm1d(INPUT_VECTOR_SIZE),  # 归一化
    nn.Linear(INPUT_VECTOR_SIZE, 64),  # 全连接层
    nn.BatchNorm1d(64),
    nn.ReLU(),  # 激活函数
    nn.Linear(64, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128, N_CLASSES),
    nn.Softmax(dim=1),  # 分类任务最后用softmax层
)

model = MLP.to(DEVICE)  # 使用BP模型
# print(model)
model.load_state_dict(load_model(FILE_PATH, DEVICE))  # 加载模型


def predict_raw_input(x):
    """预测,输入为原始数据，直接入模型"""
    model.eval()  # 验证模式
    with torch.no_grad():
        output = model(x)
        return output


def write_to_file(uuid, features):
    with open(f"./file_output/{uuid}/extracted_features.txt", "w") as f:
        for l, v in features.items():
            f.write(f"{l}: {v}\n")


def predict(uuid, sample, segmentations=None):
    features = calc_features(sample, segmentations)  # 计算特征
    if FILE_OUTPUT:
        write_to_file(uuid, features)
    features = list(features.values())
    features = torch.tensor([features], dtype=torch.float)  # 转换为tensor
    result = predict_raw_input(features.to(DEVICE)).squeeze()
    return result
