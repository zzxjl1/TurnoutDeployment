# 切换到当前目录
import os
import random
import time

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
    if "power" in sample:
        sample.pop("power")
    t["time_series"] = sample
    t["point_interval"] = 40
    # print(t)

    start_time = time.time()
    r = requests.post(url, json=t)
    end_time = time.time()

    response_time = end_time - start_time

    return r.json()["fault_diagnosis"], ground_truth, response_time


import threading


def show():
    while 1:
        time.sleep(1)
        if total_requests == 0:
            continue
        end_time_all = time.time()
        elapsed_time_all = end_time_all - start_time_all
        # tps = total_requests / elapsed_time_all
        avg_response_time = total_response_time / total_requests

        print("=========================================")
        print(f"Total Requests: {total_requests}")
        print(f"Total Elapsed Time: {elapsed_time_all} seconds")
        # print(f"TPS (Transactions Per Second): {tps}")
        print(f"Average Response Time: {avg_response_time} seconds")
        print(f"Queue Length:{len(threads)}")
        print("=========================================")


threads = []
total_requests = 0
start_time_all = time.time()
total_response_time = 0


def send_loop():
    global total_response_time
    global total_requests
    while 1:
        ran_val = random.randint(1, 10)
        print(f"Sleep for {ran_val} seconds!")
        time.sleep(ran_val)
        samples, types = get_all_samples()
        samples_to_send = random.sample(list(zip(samples, types)), ran_val)
        for sample, type in samples_to_send:
            total_requests += 1
            future = executor.submit(send, sample, type)
            # input(f"Press Enter to continue...")
            threads.append(future)

        for future in concurrent.futures.as_completed(threads):
            result = future.result()
            threads.remove(future)
            _, _, response_time = result
            total_response_time += response_time


if __name__ == "__main__":

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
    threading.Thread(target=show, daemon=True).start()
    send_loop()
