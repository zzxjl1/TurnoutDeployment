import os
import pickle
import matplotlib
from matplotlib import patches
from config import RENDER_POOL_MAX_QUQUE_SIZE
matplotlib.use("Agg")  # Use the Agg backend
import matplotlib.pyplot as plt
import functools, os
from logger_config import logger

def trace_unhandled_exceptions(func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in RENDER PROCESS:{e}")
            if isinstance(e, MemoryError):
                logger.error("Memory Error detected, restarting service...")
                import subprocess
                from utils import self_terminate
                #subprocess.Popen('run.bat', creationflags=subprocess.CREATE_NEW_CONSOLE)
                #self_terminate(flush_record=False)

    return wrapped_func

class AutoEncoderPlotter:
    fig = None

    @classmethod
    @trace_unhandled_exceptions
    def draw(cls, path, channels, y_before, y_after, ae_type, loss, series_to_encode):
        if cls.fig is None:
            logger.debug("当前进程中的可复用对象不存在，正在创建新的figure对象！")
            cls.fig, axes = plt.subplots(channels, 1, figsize=(12, 5))
        else:
            logger.debug(f"Reusing existing figure ID: {cls.fig.number}")
            cls.fig.clear()  # Clear existing content
            axes = cls.fig.subplots(channels, 1)

        for i in range(channels):
            ax = axes[i]
            ax.plot(y_before[i], label="original")
            ax.plot(y_after[i], label="AutoEncoder result")
            ax.set_title(f"Channel: {series_to_encode[i]}")
            ax.set_xlim(0, None)
            ax.set_ylim(bottom=0, top=5)

        title = f"AutoEncoder type: {ae_type} - loss: {loss}"
        cls.fig.suptitle(title)
        lines, labels = cls.fig.axes[-1].get_legend_handles_labels()
        cls.fig.legend(lines, labels, loc="upper right")
        cls.fig.set_tight_layout(True)
        cls.fig.savefig(f"{path}/{ae_type}", dpi=150)

    @classmethod
    def plot(cls, uuid, processPool):
        path = f"./file_output/{uuid}/AE"
        raw_path = f"{path}/raw"
        tasks = []
        for filename in os.listdir(raw_path):
            if filename.endswith(".pkl"):
                with open(f"{raw_path}/{filename}", "rb") as f:
                    kwargs = pickle.load(f)
                    # draw(path, **kwargs)
                    task = processPool.apply_async(
                        cls.draw, args=(path,), kwds=kwargs, error_callback=logger.exception
                    )
                    tasks.append(task)
        return tasks


class SegmentationPlotter:
    fig = None

    @classmethod
    @trace_unhandled_exceptions
    def draw(
        cls,
        path,
        duration,
        duration_index,
        gru_score,
        d2_result,
        x,
        y,
        segmentation_point_1_x,
        segmentation_point_2_x,
        name,
    ):
        if cls.fig is None:
            logger.debug("当前进程中的可复用对象不存在，正在创建新的figure对象！")
            cls.fig = plt.figure(figsize=(9, 4))
        else:
            logger.debug(f"Reusing existing figure ID: {cls.fig.number}")
            cls.fig.clear()  # Clear existing content

        ax = cls.fig.subplots()
        ax.cla()
        ax.set_xlim(0, duration)
        ax.set_yticks([])  # 不显示y轴
        ax_new = ax.twinx().twiny()
        ax_new.set_yticks([])  # 不显示y轴
        ax_new.set_xticks([])  # 不显示x轴
        ax_new.pcolormesh(
            gru_score[:duration_index].reshape(1, -1), cmap="Reds", alpha=0.7
        )
        # ax_new.plot(*model_output_to_xy(gru_score, end_sec=duration), "r")
        ax1 = ax.twinx()  # 生成第二个y轴
        ax2 = ax.twinx()  # 生成第三个y轴
        # ax2.plot(*d1_result, label="d1")
        ax2.plot(*d2_result, label="Legacy Scheme", color="red", linewidth=1, alpha=0.2)
        ax1.plot(x, y, label="Time Series", color="blue")
        ax1.set_yticks([])  # 不显示y轴
        ax2.set_yticks([])  # 不显示y轴
        # 画竖线
        if segmentation_point_1_x is not None:
            ax2.axvline(
                x=segmentation_point_1_x,
                color="r",
                linestyle="--",
                label="Segmentation Point",
            )
        if segmentation_point_2_x is not None:
            ax2.axvline(x=segmentation_point_2_x, color="r", linestyle="--")
        cls.fig.suptitle(f"Channel {name} Segmentation Result")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        heatmap_patch = patches.Rectangle((0, 0), 1, 1, fc="r", alpha=0.7)
        ax2.legend(
            lines + [heatmap_patch] + lines2,
            labels + ["GRU Score Heatmap"] + labels2,
            loc="upper right",
        )  # 显示图例
        ax.set_xlabel("Time(s)")
        cls.fig.tight_layout()
        cls.fig.savefig(f"{path}/{name}", dpi=150)

    @classmethod
    def plot(cls, uuid, processPool):
        path = f"./file_output/{uuid}/segmentations"
        raw_path = f"{path}/raw"
        tasks = []
        for filename in os.listdir(raw_path):
            if filename.endswith(".pkl"):
                with open(f"{raw_path}/{filename}", "rb") as f:
                    kwargs = pickle.load(f)
                    # draw(path, **kwargs)
                    task = processPool.apply_async(
                        cls.draw, args=(path,), kwds=kwargs, error_callback=logger.exception
                    )
                    tasks.append(task)
        return tasks


class SamplePlotter:
    fig = None

    @classmethod
    @trace_unhandled_exceptions
    def draw(cls, path, data):
        if cls.fig is None:
            logger.debug("当前进程中的可复用对象不存在，正在创建新的figure对象！")
            cls.fig = plt.figure(figsize=(9, 4))
        else:
            logger.debug(f"Reusing existing figure ID: {cls.fig.number}")
            cls.fig.clear()  # Clear existing content

        ax1 = cls.fig.subplots()
        ax1.cla()
        for phase in ["A", "B", "C"]:
            ax1.plot(*data[phase], label=f"Phase {phase}")
        cls.fig.suptitle(f"Sample after Interpolation")
        ax1.set_xlabel("Time(s)")
        ax1.set_ylabel("Current(A)")
        ax1.set_ylim(bottom=0, top=5)
        ax1.set_xlim(0, None)  # 设置x轴范围
        ax1.grid(True)
        lines, labels = ax1.get_legend_handles_labels()
        ax1.legend(lines, labels, loc="best")
        cls.fig.tight_layout()
        cls.fig.savefig(f"{path}/input_sample.png", dpi=150)

    @classmethod
    def plot(cls, uuid, processPool):
        path = f"./file_output/{uuid}"
        with open(f"{path}/input_sample.pkl", "rb") as f:
            data = pickle.load(f)
            # draw(path, data)
            task = processPool.apply_async(cls.draw, (path, data), error_callback=logger.exception)
            return task


def plot_all(uuid, processPool):
    queue_size = processPool._taskqueue.qsize()
    logger.debug(f"渲染进程池任务堆积数：{queue_size}")
    if queue_size > RENDER_POOL_MAX_QUQUE_SIZE:
        logger.error(f"当前渲染进程池发生任务堆积，触发拒绝策略！")
        raise RuntimeError("渲染进程池任务堆积！")

    ae_tasks = AutoEncoderPlotter.plot(uuid, processPool)
    input_sample = SamplePlotter.plot(uuid, processPool)
    seg_pt_tasks = SegmentationPlotter.plot(uuid, processPool)

    tasks = ae_tasks + [input_sample] + seg_pt_tasks
    for task in tasks:
        task.wait()
