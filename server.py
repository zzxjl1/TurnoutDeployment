from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import multiprocessing
from typing import Dict, Any, Union

from sample import Sample
from seg_score import GRUScore
from gru_fcn import GRU_FCN, Vanilla_GRU, FCN_1D, Squeeze_Excite
from auto_encoder import BP_AE
from mlp import MLP
from result_fusion import FuzzyLayer, FusedFuzzyDeepNet
from config import DEBUG, FILE_OUTPUT
from utils import mk_output_dir
from visualization import plot_ae, plot_sample, plot_seg_pts

app = FastAPI()


class RawData(BaseModel):
    time_series: Dict[str, list[float]]
    point_interval: Union[int, None] = 40


@app.post("/detect")
def predict(rawData: RawData):
    sample = Sample(rawData.time_series, rawData.point_interval)
    seg_points = sample.calc_seg_points()  # 计算分割点
    prediction = sample.predict()

    if FILE_OUTPUT:
        plot_sample(sample.uuid)
        plot_seg_pts(sample.uuid)
        plot_ae(sample.uuid)

    return {
        "uuid": sample.uuid,
        "time_series": sample.time_series,  # 插值后的数据
        "seg_points": seg_points,
        "fault_diagnosis": prediction,
        "file_output": FILE_OUTPUT,
    }


if __name__ == "__main__":
    mk_output_dir()
    # 获取 CPU 核心数量
    number_of_cores = multiprocessing.cpu_count()
    # 设置 worker 数量
    workers_num = 1 * number_of_cores + 1 if not DEBUG else 1
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=5000,
        reload=DEBUG,
        workers=workers_num,
    )
