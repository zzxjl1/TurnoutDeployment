import numpy as np
from config import TARGET_SAMPLE_RATE
from config import SUPPORTED_SAMPLE_TYPES
import scipy.interpolate


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
    interper = scipy.interpolate.interp1d(x, y, kind="linear")  # 线性插值
    time_elipsed = max(x) - min(x)  # 总时间
    x = np.linspace(min(x), max(x), round(time_elipsed * TARGET_SAMPLE_RATE))  # 插值
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
    result = result[::pooling_factor_per_time_series]
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
        # print(seg_index)

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