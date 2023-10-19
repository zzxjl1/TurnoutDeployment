from segmentation import calc_segmentation_points
import gru_fcn
import mlp
import auto_encoder
import result_fusion
import numpy as np
from utils import interpolate
import matplotlib

matplotlib.use("Agg")  # Use the Agg backend
import matplotlib.pyplot as plt
from config import FILE_OUTPUT, TARGET_SAMPLE_RATE
from utils import parse_predict_result


class Sample:
    def __init__(self, data, point_interval) -> None:
        self.raw_data = data
        self.data = {}
        self.point_count = 0
        self.point_interval = point_interval
        self.sample_rate = 1000 / point_interval

        self.validate()

        channel_names = ["A", "B", "C"]
        for channel_name in channel_names:
            channel_data = self.raw_data[channel_name]
            self.data[channel_name] = self.parse_channel(channel_data)

        self.plot_sample()

    @property
    def time_series(self):
        timeseries = {}
        for name, data in self.data.items():
            timeseries[name] = {
                "x": [round(t, 4) for t in data[0]],
                "y": [round(t, 4) for t in data[1]],
            }
        return timeseries

    def plot_sample(self):
        if not FILE_OUTPUT:
            return
        fig = plt.figure(dpi=150, figsize=(9, 2))
        ax1 = fig.subplots()
        for phase in ["A", "B", "C"]:
            ax1.plot(*self.data[phase], label=f"Phase {phase}")
        plt.title(f"Sample after Interpolation")
        ax1.set_xlabel("Time(s)")
        ax1.set_ylabel("Current(A)")
        ax1.set_ylim(bottom=0, top=5)
        plt.xlim(0, None)  # 设置x轴范围
        plt.grid(True)
        lines, labels = ax1.get_legend_handles_labels()
        plt.legend(lines, labels, loc="best")
        plt.savefig("./debug_output/input_sample.png")
        plt.close()

    def validate(self):
        self.point_count = len(list(self.raw_data.values())[0])
        for array in self.raw_data.values():
            assert self.point_count == len(array), "通道长度不一致"

    def parse_channel(self, channel_data):
        POINT_INTERVAL = self.point_interval / 1000
        DURATION = POINT_INTERVAL * self.point_count  # 时长

        x = np.linspace(0, DURATION, self.point_count)  # 生成x轴数据
        y = [float(i) / 100 for i in channel_data]  # y全部元素除以100
        assert len(x) == len(y)  # x和y长度相等

        return interpolate(x, y)  # 插值

    def calc_seg_points(self):
        def to_time(t):
            return round(t, 2) if t is not None else None

        def to_idx(t):
            return round(t * TARGET_SAMPLE_RATE) if t is not None else None

        def to_raw_idx(t):
            return round(t * self.sample_rate) if t is not None else None

        assert self.data is not None

        t1, t2 = calc_segmentation_points(self.data)
        self.seg_points = {
            "pt_1": {
                "time": to_time(t1),
                "raw_idx": to_raw_idx(t1),
                "index": to_idx(t1),
            },
            "pt_2": {
                "time": to_time(t2),
                "raw_idx": to_raw_idx(t2),
                "index": to_idx(t2),
            },
        }
        return self.seg_points

    def predict(self):
        seg_pts = [pt["time"] for pt in self.seg_points.values()]

        mlp_result = mlp.predict(self.data, seg_pts)
        gru_fcn_result = gru_fcn.predict(self.data)
        ae_result = auto_encoder.predict(self.data)

        final_result = result_fusion.predict(mlp_result, gru_fcn_result, ae_result)

        return {
            "sub_models": {
                "mlp": parse_predict_result(mlp_result),
                "gru_fcn": parse_predict_result(gru_fcn_result),
                "ae": parse_predict_result(ae_result),
            },
            "fusion": parse_predict_result(final_result),
        }
