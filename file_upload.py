import os
from minio import Minio
from config import ENDPOINT, ACCESS_KEY, SECRET_KEY, BUCKET_NAME, UPLOAD_MAX_WORKERS,UPLOAD
import concurrent.futures
from logger_config import logger

class FigureUploader:
    def __init__(self):
        if not UPLOAD:
            return
        self.client = Minio(
            ENDPOINT,
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            secure=False,
        )
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=UPLOAD_MAX_WORKERS,
            thread_name_prefix="uploader",
        )

        found = self.client.bucket_exists(BUCKET_NAME)
        if not found:
            self.client.make_bucket(BUCKET_NAME)

    def upload(self, file_path):
        obj_name = file_path.replace("./file_output/", "")
        logger.debug(f"开始上传{obj_name}...")
        self.client.fput_object(BUCKET_NAME, obj_name.replace("\\", "/"), file_path)

    def get_file_list(self, uuid):
        path = f"./file_output/{uuid}"
        file_list = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".pkl"):
                    continue
                file_path = os.path.join(root, file)
                file_list.append(file_path)
        return file_list

    def upload_all(self, uuid):
        file_list = self.get_file_list(uuid)
        logger.debug(f"{uuid}待上传文件列表：{file_list}")
        tasks = [self.executor.submit(self.upload, file) for file in file_list]
        # 等待所有任务完成
        concurrent.futures.wait(tasks)


if __name__ == "__main__":
    import time

    cycle = 10
    uploader = FigureUploader()
    for i in range(cycle):
        start = time.time()
        uploader.upload_all("3726bebc-32bf-4bb0-935e-28cf8b9040d6")
        end = time.time()

        print(f"第{i}次，耗时{end-start}秒")
