"""
从sample（共4条曲线）中提取特征
"""

import numpy as np
from segmentation import calc_segmentation_points
from utils import find_nearest
from collections import OrderedDict
from seg_score import GRUScore

SERIES_TO_ENCODE = ["A", "B", "C"]
IGNORE_LIST = []


def calc_features_per_stage(x, y, series_name, stage_name):
    """计算单个stage的特征"""

    result = OrderedDict()

    if x is not None and y is not None:
        result["time_span"] = x[-1] - x[0]  # 时间跨度
        result["max"] = max(y)  # 最大值
        result["min"] = min(y)  # 最小值
        result["mean"] = sum(y) / len(y)  # 平均值
        result["mean_abs"] = sum([abs(i) for i in y]) / len(y)  # 平均绝对值
        result["median"] = sorted(y)[len(y) // 2]  # 中位数
        result["std"] = (
            sum([(i - result["mean"]) ** 2 for i in y]) / len(y)
        ) ** 0.5  # Standard deviation
        result["rms"] = (sum([i**2 for i in y]) / len(y)) ** 0.5  # 均方根
        result["peak_to_peak"] = max(y) - min(y)  # 峰峰值
        try:
            result["skewness"] = sum(
                [((i - result["mean"]) / result["std"]) ** 3 for i in y]
            ) / len(y)
            result["kurtosis"] = sum(
                [((i - result["mean"]) / result["std"]) ** 4 for i in y]
            ) / len(
                y
            )  # 峭度
        except ZeroDivisionError:
            result["skewness"] = 0
            result["kurtosis"] = 0

        try:
            result["impluse_factor"] = max(y) / result["mean_abs"]
            result["form_factor"] = result["rms"] / result["mean_abs"]
        except ZeroDivisionError:
            result["impluse_factor"] = 0
            result["form_factor"] = 0

        try:
            result["crest_factor"] = max(y) / result["rms"]
        except ZeroDivisionError:
            result["crest_factor"] = 0

        try:
            result["clearance_factor"] = (
                max(y) / (sum([abs(i) ** 0.5 for i in y]) / len(y)) ** 2
            )
        except ZeroDivisionError:
            result["clearance_factor"] = 0

    else:
        result["time_span"] = -1
        result["max"] = -1
        result["min"] = -1
        result["mean"] = -1
        result["mean_abs"] = -1
        result["median"] = -1
        result["std"] = -1
        result["rms"] = -1
        result["peak_to_peak"] = -1
        result["skewness"] = -1
        result["kurtosis"] = -1

        result["impluse_factor"] = -1
        result["form_factor"] = -1
        result["crest_factor"] = -1
        result["clearance_factor"] = -1

    # nan值置0
    for k, v in result.items():
        if np.isnan(v):
            result[k] = 0

    # 所有的key前面加上series_name和stage_name
    result = OrderedDict(
        [(f"{series_name}_{stage_name}_{k}", v) for k, v in result.items()]
    )

    assert len(result) == 15  # 断言确保15个特征
    return result


def calc_features_single_series(x, y, segmentation_points, series_name):
    """
    计算单个曲线的（见下方列出）种特征
    （stage时间跨度、最大值、最小值、平均值、中位数、Standard deviation、Peak factor、Fluctuation factor）*3个stage
    """
    features = OrderedDict()

    segmentation_points_count = len(
        [i for i in segmentation_points if i is not None]
    )  # 计算划分点数

    x, y = map(lambda x: list(x), (x, y))

    if segmentation_points_count == 2:  # 2个划分点
        stage1_start, stage1_end = 0, find_nearest(
            x, segmentation_points[0]
        )  # 第一个stage的起始点和终止点索引
        stage1 = calc_features_per_stage(
            x[stage1_start:stage1_end],
            y[stage1_start:stage1_end],
            series_name,
            "stage1",
        )  # 计算第一个stage的特征
        features.update(stage1)  # 合并

        stage2_start, stage2_end = stage1_end, find_nearest(
            x, segmentation_points[1]
        )  # 第二个stage的起始点和终止点索引
        stage2 = calc_features_per_stage(
            x[stage2_start:stage2_end],
            y[stage2_start:stage2_end],
            series_name,
            "stage2",
        )  # 计算第二个stage的特征
        features.update(stage2)  # 合并

        stage3_start, stage3_end = stage2_end, len(x)  # 第三个stage的起始点和终止点索引
        stage3 = calc_features_per_stage(
            x[stage3_start:stage3_end],
            y[stage3_start:stage3_end],
            series_name,
            "stage3",
        )  # 计算第三个stage的特征
        features.update(stage3)  # 合并

    elif segmentation_points_count == 1:  # 1个划分点
        assert segmentation_points[0] is not None
        stage1_start, stage1_end = 0, find_nearest(x, segmentation_points[0])
        stage1 = calc_features_per_stage(
            x[stage1_start:stage1_end],
            y[stage1_start:stage1_end],
            series_name,
            "stage1",
        )
        features.update(stage1)

        stage2_start, stage2_end = stage1_end, len(x)
        stage2 = calc_features_per_stage(
            x[stage2_start:stage2_end],
            y[stage2_start:stage2_end],
            series_name,
            "stage2",
        )
        features.update(stage2)

        stage3 = calc_features_per_stage(None, None, series_name, "stage3")
        features.update(stage3)

    elif segmentation_points_count == 0:  # 0个划分点
        stage1_start, stage1_end = 0, len(x)
        stage1 = calc_features_per_stage(
            x[stage1_start:stage1_end],
            y[stage1_start:stage1_end],
            series_name,
            "stage1",
        )
        features.update(stage1)

        stage2 = calc_features_per_stage(None, None, series_name, "stage2")
        features.update(stage2)

        stage3 = calc_features_per_stage(None, None, series_name, "stage3")
        features.update(stage3)

    assert len(features) == 15 * 3

    return features


def calc_features(sample, segmentations=None):
    """
    计算整个sample（3个曲线*3个stage）的特征
    （stage时间跨度、最大值、最小值、平均值、中位数、Standard deviation、Peak factor、Fluctuation factor）*4个曲线*3个stage
    """
    if not segmentations:
        segmentations = calc_segmentation_points(sample)
    features = OrderedDict()
    for name, series in sample.items():  # 遍历3个曲线
        if name not in SERIES_TO_ENCODE:
            continue
        x, y = series
        total_time_elipsed = x[-1] - x[0]  # 总用时
        t = calc_features_single_series(x, y, segmentations, name)
        features.update(t)  # 合并
    features["total_time_elipsed"] = total_time_elipsed

    assert len(features) == 1 + 15 * len(SERIES_TO_ENCODE) * 3  # 断言确保特征数

    for i in IGNORE_LIST:
        features.pop(i)

    return features
