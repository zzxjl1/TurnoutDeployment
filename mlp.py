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
from config import SUPPORTED_SAMPLE_TYPES, DEBUG

FILE_PATH = "./models/mlp_classification.pth"
FORCE_CPU = True  # 强制使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
print("Using device:", DEVICE)

INPUT_VECTOR_SIZE = 1 + 15 * len(SERIES_TO_ENCODE) * 3 - len(IGNORE_LIST)  # 输入向量的大小
N_CLASSES = len(SUPPORTED_SAMPLE_TYPES)  # 分类数


class MLP(nn.Module):
    def __init__(self, input_vector_size, output_vector_size):
        super(MLP, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_vector_size)
        self.fc1 = nn.Linear(input_vector_size, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, output_vector_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        bn1_result = self.bn1(x)

        fc1_result = self.fc1(bn1_result)
        bn2_result = self.bn2(fc1_result)
        x = F.relu(bn2_result)

        fc2_result = self.fc2(x)
        bn3_result = self.bn3(fc2_result)
        x = F.relu(bn3_result)

        out = self.out(x)
        softmax_result = self.softmax(out)
        return softmax_result


model = MLP(input_vector_size=INPUT_VECTOR_SIZE, output_vector_size=N_CLASSES).to(
    DEVICE
)  # 使用BP模型
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
    if DEBUG:
        write_to_file(features)
    features = list(features.values())
    features = torch.tensor([features], dtype=torch.float)  # 转换为tensor
    result = predict_raw_input(features.to(DEVICE)).squeeze()
    return result
