"""
读取excel表格中的现实世界数据并打包为sample
"""
import time
import numpy as np
import pandas as pd
from enum import Enum
import requests

column_names = [
    "timestamp",
    "direction",
    "data",
    "point_count",
    "current_type",
    "curve_type",
    "turnout_name",
]  # 列名
df = pd.read_excel("./test/turnoutActionCurve.xlsx")  # 读取excel文件


class CurveType(Enum):
    A = 1
    B = 2
    C = 3
    power = 4


class CurrentType(Enum):
    AC = 0
    DC = 1


def read_row(i):
    # 获取第i行的数据
    row_data = df.iloc[i, :]
    # 映射到column_names
    result = dict(zip(column_names, row_data.values))
    return result


seq = ["A", "B", "C", "power"]  # 曲线顺序


def validate(i):
    for j in range(4):  # 遍历四个曲线
        t = read_row(i + j)  # 获取第i+j行的数据
        if t["curve_type"] != CurveType[seq[j]].value:  # 顺序不匹配
            return False

        # if t["current_type"] != CurrentType.DC.value:  # 电流类型不匹配
        #    return False

        return True


def parse(i):
    result = {}
    for j in range(4):  # 遍历四个曲线
        t = read_row(i + j)  # 获取第i+j行的数据
        type = CurveType(t["curve_type"]).name  # 曲线类型
        result[type] = t["data"].split(",")  # 解析数据
        result[type] = list(map(float, result[type]))  # 全部元素除以100

    # print(result)
    # show_sample(result)
    return result


def get_all_samples():
    result = []
    row = df.shape[0]  # 总行数
    i = 0
    while i < row:  # 遍历每一行
        if not validate(i):  # 过滤非法数据
            i += 1
            print(f"line {i} validate failed")
            continue

        print(f"line {i} parsed")
        result.append(parse(i))  # 解析数据后添加到result
        i += 4

    print(len(result))
    return result


def send(sample, concurrency=1):
    def do():
        nonlocal count
        start_time = time.time()  # Record the start time
        r = requests.post(url, json=t)
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")
        count += 1
        print(f"count: {count},total:{concurrency}")

        # with open("./file_output/response.txt", "w") as f:
        #    f.write(r.text)

    url = "http://localhost:5000/detect"
    t = {}
    sample.pop("power")
    t["time_series"] = sample
    t["point_interval"] = 40
    print(t)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        threads = []
        count = 0
        for _ in range(concurrency):
            future = executor.submit(do)
            threads.append(future)

        for thread in threads:
            thread.result()
        print("all done")


if __name__ == "__main__":
    import os
    import sys
    import requests
    import concurrent.futures

    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parentdir)

    for sample in get_all_samples():
        print(sample)
        send(sample, 5)
        input()
