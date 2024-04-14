import psutil
import time
from datetime import datetime

def record_memory_usage():
    while True:
        # 获取当前时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 获取系统内存占用情况
        memory = psutil.virtual_memory()
        total_memory = memory.total
        available_memory = memory.available
        used_memory = memory.used
        memory_percent = memory.percent

        # 将内存占用情况和时间写入文件（追加模式）
        with open("./memory_usage.log", "a") as f:
            f.write(f"时间: {current_time}\n")
            f.write(f"总内存: {total_memory} bytes\n")
            f.write(f"可用内存: {available_memory} bytes\n")
            f.write(f"已用内存: {used_memory} bytes\n")
            f.write(f"内存使用率: {memory_percent}%\n")
            f.write("-" * 40 + "\n")  # 分隔符

        # 每秒记录一次
        time.sleep(1)

if __name__ == "__main__":
    record_memory_usage()