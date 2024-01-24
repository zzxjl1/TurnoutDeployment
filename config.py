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
DEBUG = False  # 是否开启web server调试模式

""" 性能参数配置 """
FORCE_CPU = True  # 强制使用CPU跑模型
N_WORKERS = 2  # web server worker数量（-1为自动检测）
RENDER_POOL_SIZE = 3  # 渲染进程池大小（per worker）
CONCURRENCY_LIMIT = None  # 服务最大并发数（None为不限制）,超过直接返回503

""" 文件对象存储配置 """
UPLOAD = False  # 是否上传文件
BUCKET_NAME = "turnout-diagnosis"  # 存储桶名称
ENDPOINT = "127.0.0.1:9000"
ACCESS_KEY = "o27CAfziq7ww6HmYzxhF"
SECRET_KEY = "oaRxky2HzDoBmpA7yW9iPNPc2ESf63SpaMyh3k0E"
UPLOAD_MAX_WORKERS = 3  # 文件上传最大线程数（per uploader）
TASK_FINISH_CALLBACK = True  # 是否在上传完成后回调
DELETE_AFTER_UPLOAD = True  # 是否在上传完成后删除本地文件
CALLBACK_URL = "http://localhost:5000/callback"  # 完成后的回调地址（GET请求，参数为?uuid=xxxxx）
