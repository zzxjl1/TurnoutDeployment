"""
pytorch implementation of dnn classification
用dnn对提取的特征进行分类,代替原文中的svm
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from extract_features import IGNORE_LIST, calc_features, SERIES_TO_ENCODE
from seg_score import GRUScore
from utils import get_label_from_result_pretty, parse_predict_result
from config import SUPPORTED_SAMPLE_TYPES, FILE_OUTPUT, FORCE_CPU

FILE_PATH = "./models/mlp_classification.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
print("Using device:", DEVICE)

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
print(model)


def predict_raw_input(x):
    """预测,输入为原始数据，直接入模型"""
    assert os.path.exists(FILE_PATH), "model not found, please run train() first!"
    model = torch.load(FILE_PATH, map_location=DEVICE).to(DEVICE)  # 加载模型
    model.eval()  # 验证模式
    with torch.no_grad():
        output = model(x)
        return output


def write_to_file(features):
    with open("./debug_output/extracted_features.txt", "w") as f:
        for l, v in features.items():
            f.write(f"{l}: {v}\n")


def predict(sample, segmentations=None):
    features = calc_features(sample, segmentations)  # 计算特征
    if FILE_OUTPUT:
        write_to_file(features)
    features = list(features.values())
    features = torch.tensor([features], dtype=torch.float)  # 转换为tensor
    result = predict_raw_input(features.to(DEVICE)).squeeze()
    return result
