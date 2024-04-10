import multiprocessing
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from logger_config import logger
from config import (
    FILE_OUTPUT,
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


logger.debug(f"--- DEBUG INFO ---")
logger.debug(f"当前进程: {multiprocessing.current_process()}")
logger.debug(f"是否渲染进程: {is_render_process()}")
logger.debug(f"是否Web服务器进程: {not is_render_process()}")
logger.debug(f"是否主进程: {is_main_process()}")


from fastapi import FastAPI
from fastapi.responses import JSONResponse
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
    from utils import mk_output_dir, record_pid, self_terminate
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
    record_pid()
    mk_output_dir()
    global renderProcessPool
    renderProcessPool = multiprocessing.get_context("spawn").Pool(
        processes=RENDER_POOL_SIZE,
        maxtasksperchild=RENDER_POOL_MAX_TASKS_PER_PROC
    )
    global backgroundTasksPool
    backgroundTasksPool = ThreadPoolExecutor(max_workers=RENDER_POOL_SIZE)
    logger.info(f"当前web服务器进程: {multiprocessing.current_process().name}")
    logger.info(f"对应渲染进程池为: {renderProcessPool}")
    logger.info(f"后台任务线程池为: {backgroundTasksPool}")


@app.on_event("shutdown")
def shutdown_event():
    renderProcessPool.terminate()
    renderProcessPool.close()
    renderProcessPool.join()
    logger.info("渲染进程池已关闭！")
    backgroundTasksPool.shutdown()
    logger.info("后台任务线程池已关闭！")
    self_terminate()

# 注册全局异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception("An unhandled exception occurred: %s", exc)
    return JSONResponse(status_code=500, content={"message": str(exc)})


def send_callback(uuid: str):
    if not TASK_FINISH_CALLBACK:
        return
    requests.get(f"{CALLBACK_URL}?uuid={uuid}")


def del_after_upload(uuid: str):
    if not DELETE_AFTER_UPLOAD:
        return
    shutil.rmtree(f"./file_output/{uuid}")
    logger.info(f"{uuid}的本地文件已删除！")


def plot_and_upload(uuid: str):
    logger.info(f"开始生成{uuid}的可视化文件...")
    visualization.plot_all(uuid, renderProcessPool)
    logger.info(f"{uuid}生成已全部完成！")
    # del_after_upload(uuid) # DEBUG ONLY

    if not UPLOAD:
        return

    logger.info(f"开始上传{uuid}...")
    minioUploader.upload_all(uuid)
    logger.info(f"上传完成：{uuid}")
    send_callback(uuid)
    del_after_upload(uuid)


class RawData(BaseModel):
    time_series: Dict[str, list[float]]
    point_interval: Union[int, None] = 40


@app.post("/detect")
async def predict(rawData: RawData):
    logger.debug(f"收到原始序列: {rawData}")
    sample = Sample(rawData.time_series, rawData.point_interval)
    seg_points = sample.calc_seg_points()  # 计算分割点
    logger.debug(f"分割点: {seg_points}")
    prediction = sample.predict()
    logger.debug(f"预测结果: {prediction}")

    if FILE_OUTPUT:
        # background_tasks.add_task(plot_and_upload, sample.uuid)
        # threading.Thread(target=plot_and_upload, args=(sample.uuid,)).start()
        queue_size = backgroundTasksPool._work_queue.qsize()
        logger.debug(f"当前后台任务线程池任务堆积数：{queue_size}")
        if queue_size > MAX_BG_TASKS:
            logger.error(f"当前后台任务线程池发生任务堆积，触发拒绝策略！")
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
    logger.info(f"收到{uuid}的成功回调！")
    return

@app.get("/force_restart")
async def force_restart():
    logger.warning("收到来自remote的强制重启服务请求！")
    # os.system("run.bat")
    subprocess.Popen('run.bat', creationflags=subprocess.CREATE_NEW_CONSOLE)
    self_terminate(flush_record=False)

def deamon():
    from config import MAX_MEM_USAGE_IN_GB
    from utils import get_total_memory_usage
    logger.info("Deamon thread started!")
    while True:
        time.sleep(10)
        usage = get_total_memory_usage()
        logger.info(f"当前内存占用：{round(usage/1024/1024/1024,2)}GB")
        if usage > MAX_MEM_USAGE_IN_GB*1024*1024*1024:
            logger.error("内存占用超限，服务正在重启！")
            time.sleep(2)
            # os.system("run.bat")
            subprocess.Popen('run.bat', creationflags=subprocess.CREATE_NEW_CONSOLE)
            self_terminate(flush_record=False)
            break
        

if __name__ == "__main__":
    import uvicorn
    import time
    import threading
    from utils import get_workers_num
    from utils import get_console_title
    from utils import flush_pid
    
    logger.info("正在清理上一次运行残留的PID记录...")
    self_terminate()
    flush_pid()
    record_pid()
    threading.Thread(target=deamon).start()

    logger.info(f"Current working directory: {script_dir}")
    if get_console_title() != "RailwayTurnoutGuard":
        raise RuntimeError("错误的启动方式，请双击 run.bat 运行!")

    uvicorn.run(
        "server:app",  # Use the import string of the class
        host=HOST,
        port=PORT,
        workers=get_workers_num(),
        limit_concurrency=CONCURRENCY_LIMIT,
    )
