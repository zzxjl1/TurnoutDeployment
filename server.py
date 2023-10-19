from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any, Union

from sample import Sample
from seg_score import GRUScore
from gru_fcn import GRU_FCN, Vanilla_GRU, FCN_1D, Squeeze_Excite
from auto_encoder import BP_AE
from mlp import MLP
from result_fusion import FuzzyLayer, FusedFuzzyDeepNet
from config import DEBUG


app = FastAPI()


class RawData(BaseModel):
    time_series: Dict[str, list[float]]
    point_interval: Union[int, None] = 40


@app.post("/detect")
def api(rawData: RawData):
    print(rawData)
    sample = Sample(rawData.time_series, rawData.point_interval)
    seg_points = sample.calc_seg_points()  # 计算分割点
    prediction = sample.predict()

    return {
        "time_series": sample.time_series,  # 插值后的数据
        "seg_points": seg_points,
        "fault_diagnosis": prediction,
    }


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=5000,
        debug=DEBUG,
        workers=1 if DEBUG else 4,
    )
