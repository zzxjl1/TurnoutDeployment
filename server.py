import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from config import (
    DEBUG,
    FILE_OUTPUT,
    FORCE_CPU,
    RENDER_POOL_SIZE,
    CALLBACK_URL,
    TASK_FINISH_CALLBACK,
    CONCURRENCY_LIMIT,
    UPLOAD,
    HOST,
    PORT,
    DELETE_AFTER_UPLOAD,
    RENDER_POOL_MAX_TASKS_PER_PROC,
    MAX_BG_TASKS
)

# 获取当前脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 切换工作目录到当前脚本所在目录
os.chdir(script_dir)


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


from fastapi import FastAPI
from pydantic import BaseModel
import visualization
from typing import Dict, Any, Union

if not is_render_process():
    from sample import Sample
    from seg_score import GRUScore
    from gru_fcn import GRU_FCN, Vanilla_GRU, FCN_1D, Squeeze_Excite
    from auto_encoder import BP_AE
    from mlp import MLP
    from result_fusion import FuzzyLayer, FusedFuzzyDeepNet
    from utils import mk_output_dir
    from file_upload import FigureUploader
    import requests
    import shutil

    minioUploader = FigureUploader()


app = FastAPI()


@app.get("/")
async def index():
    return "Railway Turnout Guard V2.0"


@app.on_event("startup")
def startup_event():
    mk_output_dir()
    global renderProcessPool
    renderProcessPool = multiprocessing.get_context("spawn").Pool(
        processes=RENDER_POOL_SIZE,
        maxtasksperchild=RENDER_POOL_MAX_TASKS_PER_PROC
    )
    global backgroundTasksPool
    backgroundTasksPool = ThreadPoolExecutor(max_workers=RENDER_POOL_SIZE)
    print(f"当前web服务器进程: {multiprocessing.current_process().name}")
    print(f"对应渲染进程池为: {renderProcessPool}")
    print(f"后台任务线程池为: {backgroundTasksPool}")


@app.on_event("shutdown")
def shutdown_event():
    renderProcessPool.terminate()
    renderProcessPool.close()
    renderProcessPool.join()
    print("渲染进程池已关闭！")
    backgroundTasksPool.shutdown()
    print("后台任务线程池已关闭！")


def send_callback(uuid: str):
    if not TASK_FINISH_CALLBACK:
        return
    requests.get(f"{CALLBACK_URL}?uuid={uuid}")


def del_after_upload(uuid: str):
    if not DELETE_AFTER_UPLOAD:
        return
    shutil.rmtree(f"./file_output/{uuid}")
    print(f"{uuid}的本地文件已删除！")


def plot_and_upload(uuid: str):
    print(f"开始生成{uuid}的可视化文件...")
    visualization.plot_all(uuid, renderProcessPool)
    print(f"{uuid}生成已全部完成！")
    # del_after_upload(uuid) # DEBUG ONLY

    if not UPLOAD:
        return

    print(f"开始上传{uuid}...")
    minioUploader.upload_all(uuid)
    print(f"上传完成：{uuid}")
    send_callback(uuid)
    del_after_upload(uuid)


class RawData(BaseModel):
    time_series: Dict[str, list[float]]
    point_interval: Union[int, None] = 40


@app.post("/detect")
async def predict(rawData: RawData):
    print("收到原始序列:", rawData)
    sample = Sample(rawData.time_series, rawData.point_interval)
    seg_points = sample.calc_seg_points()  # 计算分割点
    print("分割点:", seg_points)
    prediction = sample.predict()
    print("预测结果:", prediction)

    if FILE_OUTPUT:
        # background_tasks.add_task(plot_and_upload, sample.uuid)
        # threading.Thread(target=plot_and_upload, args=(sample.uuid,)).start()
        queue_size = backgroundTasksPool._work_queue.qsize()
        if DEBUG:
            print(f"当前后台任务线程池任务堆积数：{queue_size}")
        if queue_size > MAX_BG_TASKS:
            print(f"当前后台任务线程池发生任务堆积，触发拒绝策略！")
            raise RuntimeError("后台任务线程池任务堆积！")
        backgroundTasksPool.submit(plot_and_upload, sample.uuid)

    return {
        "uuid": sample.uuid,
        "time_series": sample.time_series,  # 插值后的数据
        "seg_points": seg_points,
        "fault_diagnosis": prediction,
        "file_output": FILE_OUTPUT,
    }


@app.get("/callback")
async def callback(uuid: str):
    print(f"收到{uuid}的成功回调！")
    return

@app.get("/force_restart")
async def force_restart():
    print("收到来自remote的强制重启服务请求！")
    os.system("run.bat")
    return 


if __name__ == "__main__":
    import uvicorn
    import time
    from utils import get_workers_num, get_console_title

    print("Current working directory:", script_dir)
    if get_console_title() != "RailwayTurnoutGuard":
        print("错误的启动方式，请双击 run.bat 运行!")
        time.sleep(5)
        raise RuntimeError("错误的启动方式")

    uvicorn.run(
        "server:app",  # Use the import string of the class
        host=HOST,
        port=PORT,
        # reload=DEBUG,
        workers=get_workers_num(),
        limit_concurrency=CONCURRENCY_LIMIT,
    )
