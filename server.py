import multiprocessing
from config import DEBUG, FILE_OUTPUT, FORCE_CPU


def is_main_process():
    return "MainProcess" in multiprocessing.current_process().name


def is_render_process():
    return "PoolWorker" in multiprocessing.current_process().name


if DEBUG:
    import torch

    print("---DEBUG MODE---")
    print("当前进程:", multiprocessing.current_process())
    print("是否渲染进程:", is_render_process())
    print("是否Web服务器进程:", not is_render_process())
    print("是否主进程:", is_main_process())
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu"
    )
    print("Using device:", DEVICE)


from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from visualization import plot_all
from typing import Dict, Any, Union

if not is_render_process():

    from sample import Sample
    from seg_score import GRUScore
    from gru_fcn import GRU_FCN, Vanilla_GRU, FCN_1D, Squeeze_Excite
    from auto_encoder import BP_AE
    from mlp import MLP
    from result_fusion import FuzzyLayer, FusedFuzzyDeepNet

    from pydantic import BaseModel
    from utils import mk_output_dir
    from config import DEBUG, FILE_OUTPUT


app = FastAPI()


class RawData(BaseModel):
    time_series: Dict[str, list[float]]
    point_interval: Union[int, None] = 40


@app.on_event("startup")
def startup_event():
    mk_output_dir()
    global renderProcessPool
    renderProcessPool = multiprocessing.Pool(processes=2)
    print(renderProcessPool)


@app.on_event("shutdown")
def shutdown_event():
    renderProcessPool.close()
    renderProcessPool.terminate()
    renderProcessPool.join()


@app.post("/detect")
def predict(rawData: RawData, background_tasks: BackgroundTasks):
    sample = Sample(rawData.time_series, rawData.point_interval)
    seg_points = sample.calc_seg_points()  # 计算分割点
    prediction = sample.predict()

    if FILE_OUTPUT:
        # background_tasks.add_task(plot_all, sample.uuid)
        renderProcessPool.apply_async(plot_all, (sample.uuid,))

    return {
        "uuid": sample.uuid,
        "time_series": sample.time_series,  # 插值后的数据
        "seg_points": seg_points,
        "fault_diagnosis": prediction,
        "file_output": FILE_OUTPUT,
    }


if __name__ == "__main__":
    import uvicorn

    number_of_cores = multiprocessing.cpu_count()
    print("CPU 核心数量: ", number_of_cores)
    workers_num = number_of_cores // 2 if not DEBUG else 1
    print("HTTP Server worker 数量设置为: ", workers_num)

    uvicorn.run(
        "server:app",  # Use the import string of the class
        host="0.0.0.0",
        port=5000,
        reload=DEBUG,
        workers=workers_num,
        # limit_concurrency=workers_num // 2, # 限制并发
    )
