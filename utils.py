import os
import numpy as np
import torch
from config import TARGET_SAMPLE_RATE
from config import SUPPORTED_SAMPLE_TYPES
from config import N_WORKERS
import scipy.interpolate
from scipy import signal
from logger_config import logger


def find_nearest(array, value):
    """找到最近的点，返回索引"""
    if value is None:
        return None
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def generate_power_series(current_series, power_factor=0.8):
    """
    产生瓦数曲线，采用直接计算的方式，需传入三项电流曲线
    P (kW) = I (Amps) × V (Volts) × PF(功率因数) × 1.732
    """
    # x, _ = current_series['A']
    # 取三相电流曲线最长的作为功率曲线x轴
    x = []
    for series in current_series.values():
        t, _ = series
        if len(t) > len(x):
            x = t
    length = len(x)
    result = np.zeros(length)
    for phase in ["A", "B", "C"]:
        for i in range(length):
            _, current = current_series[phase]
            result[i] += (
                current[i] * 220 * power_factor * 1.732 if i < len(current) else 0
            )
    return x, result


def interpolate(x, y):
    """根据关键点插值到固定采样率"""
    time_elipsed = max(x) - min(x)  # 总时间
    target_length = round(time_elipsed * TARGET_SAMPLE_RATE)
    if len(x) == target_length:
        return x, y

    interper = scipy.interpolate.interp1d(x, y, kind="linear")  # 线性插值
    x = np.linspace(min(x), max(x), target_length)  # 插值
    y = interper(x)
    return x, y


def parse_time_series(time_series, time_series_length, pooling_factor_per_time_series):
    # 超长的截断，短的补0, 之后再降采样
    if len(time_series) > time_series_length:
        result = np.array(time_series[:time_series_length])
    else:
        result = np.pad(
            time_series, (0, time_series_length - len(time_series)), "constant"
        )
    # result = result[::pooling_factor_per_time_series]
    result = signal.decimate(result, pooling_factor_per_time_series)
    return result


def parse_sample(
    sample,
    segmentations,
    time_series_length,
    pooling_factor_per_time_series,
    series_to_encode,
):
    time_series = []
    seg_index = []
    for name in series_to_encode:
        x, y = sample[name][0], sample[name][1]
        result = parse_time_series(
            y, time_series_length, pooling_factor_per_time_series
        )
        x = parse_time_series(x, time_series_length, pooling_factor_per_time_series)
        assert len(x) == len(result)
        time_series.append(result)
        if segmentations is not None:
            seg_index = [find_nearest(x, seg) for seg in segmentations]
        else:
            seg_index = None
        # logger.info(seg_index)

    result = np.array(time_series)
    return result, seg_index


def parse_predict_result(result):
    """解析预测结果"""
    result_pretty = [round(i, 2) for i in result.tolist()]
    result_pretty = dict(zip(SUPPORTED_SAMPLE_TYPES, result_pretty))  # 让输出更美观
    return result_pretty


def get_label_from_result_pretty(result_pretty):
    """从解析后的预测结果中获取标签"""
    return max(result_pretty, key=result_pretty.get)


def mk_output_dir(uuid=None):
    """创建输出目录"""
    if uuid is None:
        dirs = ["./file_output"]
    else:
        dirs = [
            f"./file_output/{uuid}",
            f"./file_output/{uuid}/AE",
            f"./file_output/{uuid}/AE/raw",
            f"./file_output/{uuid}/segmentations",
            f"./file_output/{uuid}/segmentations/raw",
        ]

    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)


import pickle


def save_args(filepath, args):
    """保存参数"""
    with open(filepath, "wb") as f:
        pickle.dump(args, f)


def get_workers_num():
    """获取web server worker数量"""
    import multiprocessing

    number_of_cores = multiprocessing.cpu_count()
    logger.info(f"CPU 核心数量: {number_of_cores}")
    workers_num = (number_of_cores-1) // 8 if N_WORKERS == -1 else N_WORKERS
    assert workers_num >= 0, "worker数量非法！"
    logger.info(f"web server worker数量设置为: {workers_num}")
    return workers_num

def load_model(FILE_PATH,DEVICE):
    assert os.path.exists(FILE_PATH), f"{FILE_PATH} not found, please train first!"
    model = torch.load(FILE_PATH, map_location=DEVICE)  # 加载模型
    logger.info(f"模型{FILE_PATH}加载成功!")
    return model

import ctypes, psutil

def set_console_title(new_title):
    ctypes.windll.kernel32.SetConsoleTitleW(new_title)

def get_console_title():
    BUF_SIZE = 256
    buffer = ctypes.create_unicode_buffer(BUF_SIZE)
    ctypes.windll.kernel32.GetConsoleTitleW(buffer, BUF_SIZE)
    logger.info(f"Window title: {buffer.value}")
    return buffer.value

PID_FILE = './pids.txt'
def flush_pid():
    try:
        os.remove(PID_FILE)  # 删除旧的PID文件
        logger.info("旧的PID记录文件已删除")
    except FileNotFoundError:
        pass
    
def record_pid():
    pid = os.getpid()  # 获取当前进程的PID
    with open(PID_FILE, 'a') as f:
        f.write(str(pid) + '\n')  # 将PID写入文件
    logger.info(f"pid {pid} recorded!")

def get_pids():
    try:
        with open(PID_FILE, 'r') as f:
            pids = [pid.strip() for pid in f.readlines()]
        return list(map(int, pids))
    except FileNotFoundError:
        logger.info("未找到PID文件")
        return []

def self_terminate():
    pids = get_pids()
    flush_pid()
    logger.info(f"记录中的pid：{pids}")
    for pid in pids:
        try:
            logger.info(f"尝试结束进程{pid}")
            psutil.Process(pid).kill()
            logger.info(f"进程{pid}已结束")
        except psutil.NoSuchProcess:
            logger.info(f"进程{pid}不存在")
        except Exception as e:
            logger.error(f"结束进程{pid}失败: {e}")

def get_total_memory_usage():
    res = 0
    pids = set()
    for pid in get_pids():
        try:
            pids.add(pid)
            p = psutil.Process(pid)
            for child in p.children(recursive=True):
                pids.add(child.pid)
        except psutil.NoSuchProcess:
            pass
    
    logger.info(f"当前进程树: {list(pids)}")
    for pid in pids:
        try:
            p = psutil.Process(pid)
            res += p.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return res
    