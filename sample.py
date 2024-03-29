from segmentation import calc_segmentation_points
import gru_fcn
import mlp
import auto_encoder
import result_fusion
import numpy as np
from utils import interpolate, mk_output_dir, save_args


from config import FILE_OUTPUT, TARGET_SAMPLE_RATE
from utils import parse_predict_result
import concurrent.futures
import uuid

# import pysnooper


class Sample:
    def __init__(self, data, point_interval) -> None:
        self.uuid = uuid.uuid4()
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

        if FILE_OUTPUT:
            mk_output_dir(self.uuid)
            save_args(f"./file_output/{self.uuid}/input_sample.pkl", self.data)

    @property
    def time_series(self):
        timeseries = {}
        for name, data in self.data.items():
            timeseries[name] = {
                "x": [round(t, 4) for t in data[0]],
                "y": [round(t, 4) for t in data[1]],
            }
        return timeseries

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

    # @pysnooper.snoop()
    def calc_seg_points(self):
        def to_time(t):
            return round(t, 2) if t is not None else None

        def to_idx(t):
            return round(t * TARGET_SAMPLE_RATE) if t is not None else None

        def to_raw_idx(t):
            return round(t * self.sample_rate) if t is not None else None

        assert self.data is not None

        t1, t2 = calc_segmentation_points(self.uuid, self.data)
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
        print(self.seg_points)
        return self.seg_points

    # @pysnooper.snoop(thread_info=True)
    def predict(self):
        seg_pts = [pt["time"] for pt in self.seg_points.values()]

        """
        mlp_result = mlp.predict(self.uuid,self.data, seg_pts)
        gru_fcn_result = gru_fcn.predict(self.uuid,self.data)
        ae_result = auto_encoder.predict(self.uuid,self.data)
        """

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 提交任务到线程池或进程池进行并行执行
            mlp_future = executor.submit(mlp.predict, self.uuid, self.data, seg_pts)
            gru_fcn_future = executor.submit(gru_fcn.predict, self.uuid, self.data)
            ae_future = executor.submit(auto_encoder.predict, self.uuid, self.data)

            # gather结果
            mlp_result = mlp_future.result()
            print("mlp_result:", mlp_result)
            gru_fcn_result = gru_fcn_future.result()
            print("gru_fcn_result:", gru_fcn_result)
            ae_result = ae_future.result()
            print("ae_result:", ae_result)

        final_result = result_fusion.predict(mlp_result, gru_fcn_result, ae_result)

        return {
            "sub_models": {
                "mlp": parse_predict_result(mlp_result),
                "gru_fcn": parse_predict_result(gru_fcn_result),
                "ae": parse_predict_result(ae_result),
            },
            "fusion": parse_predict_result(final_result),
        }
