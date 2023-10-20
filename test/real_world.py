# 切换到当前目录
import os
import random

parentdir = os.path.dirname(os.path.abspath(__file__))
print(parentdir)
os.chdir(parentdir)

import requests
import concurrent.futures
import numpy as np
import pandas as pd
from enum import Enum

SUPPORTED_SAMPLE_TYPES = [
    "normal",
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
]

column_names = [
    "timestamp",
    "direction",
    "data",
    "point_count",
    "current_type",
    "curve_type",
    "turnout_name",
]  # 列名


class CurveType(Enum):
    A = 1
    B = 2
    C = 3
    power = 4


class CurrentType(Enum):
    AC = 0
    DC = 1


def read_row(df, i):
    # 获取第i行的数据
    row_data = df.iloc[i, :]
    # 映射到column_names
    result = dict(zip(column_names, row_data.values))
    return result


seq = ["A", "B", "C", "power"]  # 曲线顺序


def validate(df, i):
    for j in range(4):  # 遍历四个曲线
        t = read_row(df, i + j)  # 获取第i+j行的数据
        if t["curve_type"] != CurveType[seq[j]].value:  # 顺序不匹配
            return False
        """
        if t["current_type"] != CurrentType.DC.value:  # 电流类型不匹配
            return False
        """
        return True


def parse(df, i):
    result = {}
    for j in range(4):  # 遍历四个曲线
        t = read_row(df, i + j)  # 获取第i+j行的数据
        type = CurveType(t["curve_type"]).name  # 曲线类型
        result[type] = polish_data(t["data"], t["point_count"], type)
        # result[type] = list(map(float, t["data"].split(",")))  # 527条数据
    # print(result)
    # show_sample(result)
    assert len(result["A"]) == len(result["B"]) == len(result["C"])
    return result


def polish_data(data, point_count, type):
    POINT_INTERVAL = 40 / 1000  # 40ms
    DURATION = POINT_INTERVAL * point_count  # 时长

    x = np.linspace(0, DURATION, point_count)  # 生成x轴数据
    y = data.split(",")  # 生成y轴数据
    if type in ["A", "B", "C"]:
        y = [float(i) for i in y]  # y全部元素除以100
    assert len(x) == len(y)  # x和y长度相等

    # return interpolate(x, y)  # 插值
    return y


CACHE = {}


def get_samples_by_type(type="normal"):
    global CACHE
    if type in CACHE:
        return CACHE[type]
    result = []
    df = pd.read_excel(f".//{type}.xlsx")  # 读取excel文件
    print(f"read {type}.xlsx")
    row = df.shape[0]  # 总行数
    i = 0
    while i < row:  # 遍历每一行
        if not validate(df, i):  # 过滤非法数据
            i += 1
            print(f"line {i} validate failed")
            continue

        # print(f"line {i} parsed")
        try:
            temp = parse(df, i)  # 解析数据
        except:
            i += 1
            print(f"line {i} parse failed")
            continue
        result.append(temp)
        i += 4

    CACHE[type] = result
    return result


def get_all_samples(type_list=SUPPORTED_SAMPLE_TYPES):
    samples = []
    types = []
    for type in type_list:
        t = get_samples_by_type(type)
        for sample in t:
            samples.append(sample)
            types.append(type)
    return shuffle(samples, types)


def shuffle(*lists):
    """打乱多个列表"""
    l = list(zip(*lists))
    random.shuffle(l)
    return zip(*l)


def send(sample, ground_truth):
    url = "http://localhost:5000/detect"
    t = {}
    sample.pop("power")
    t["time_series"] = sample
    t["point_interval"] = 40
    # print(t)

    r = requests.post(url, json=t)
    return r.json()["fault_diagnosis"], ground_truth


def get_label_from_result_pretty(result_pretty):
    """从解析后的预测结果中获取标签"""
    return max(result_pretty, key=result_pretty.get)


if __name__ == "__main__":
    samples, types = get_all_samples()
    total = len(samples)
    print(total)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    threads = []
    for sample, type in zip(samples, types):
        future = executor.submit(send, sample, type)
        threads.append(future)

    counter = {
        "mlp": 0,
        "gru_fcn": 0,
        "ae": 0,
        "fusion": 0,
    }

    for thread in threads:
        result = thread.result()
        output, ground_truth = result
        merged = {**output["sub_models"], **{"fusion": output["fusion"]}}
        # print(merged)
        for name in counter.keys():
            counter[name] += (
                1 if get_label_from_result_pretty(merged[name]) == ground_truth else 0
            )

    print("all done")
    print(total)
    print(counter)
