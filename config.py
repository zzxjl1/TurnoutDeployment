""" 模型配置（必须与权重文件匹配，否则出错） """
SUPPORTED_SAMPLE_TYPES = [
    "normal",
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
]  # 支持的样本类型
TARGET_SAMPLE_RATE = 25  # 模型输入采样率


""" 基础配置 """
HOST = "0.0.0.0"  # web server监听地址
PORT = 5000  # 端口
FILE_OUTPUT = True  # 是否输出文件
DEBUG = False  # 是否开启调试模式

""" 性能参数配置 """
FORCE_CPU = True  # 强制使用CPU跑模型
N_WORKERS = -1  # web server worker数量（-1为自动检测）
RENDER_POOL_SIZE = 3  # 渲染进程池大小（per worker）
RENDER_POOL_MAX_TASKS_PER_PROC = None  # 渲染进程最大复用次数（None为无限次）
RENDER_POOL_MAX_QUQUE_SIZE = 50  # 渲染进程池最大任务堆积数
MAX_BG_TASKS = 100  # 最大后台任务数
CONCURRENCY_LIMIT = None  # 服务最大并发数（None为不限制）,超过直接返回503
MAX_MEM_USAGE_IN_GB = 4.0 # 最大内存占用，超过将重启服务

""" 文件对象存储配置 """
UPLOAD = False  # 是否上传文件
BUCKET_NAME = "turnout-diagnosis"  # 存储桶名称
ENDPOINT = "127.0.0.1:9000"
ACCESS_KEY = "U3UlH092S0ntcVjPAKsC"
SECRET_KEY = "LSXs3xUZs41uee3I2ZWCnnq9FDDppIIW6qFZcaFg"
UPLOAD_MAX_WORKERS = 5  # 文件上传最大线程数（per uploader）
TASK_FINISH_CALLBACK = True  # 是否在上传完成后回调
DELETE_AFTER_UPLOAD = True  # 是否在上传完成后删除本地文件
CALLBACK_URL = "http://localhost:5000/callback"  # 完成后的回调地址（GET请求，参数为?uuid=xxxxx）
