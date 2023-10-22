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

""" 性能参数配置 """
FORCE_CPU = False  # 强制使用CPU跑模型
N_WORKERS = 1  # web server worker数量（-1为自动检测）
RENDER_POOL_SIZE = 5  # 渲染进程池大小（per worker）

""" 功能配置 """
FILE_OUTPUT = True  # 是否输出文件
DEBUG = False  # 是否开启web server调试模式
TASK_FINISH_CALLBACK = True  # 是否在渲染完成后回调
CALLBACK_URL = "http://localhost:8000/callback"  # 完成后的回调地址（GET请求，参数为?uuid=xxxxx）

""" 文件对象存储配置 """
BUCKET_NAME = "turnout-diagnosis"  # 存储桶名称
ENDPOINT = "127.0.0.1:9000"
ACCESS_KEY = "o27CAfziq7ww6HmYzxhF"
SECRET_KEY = "oaRxky2HzDoBmpA7yW9iPNPc2ESf63SpaMyh3k0E"
UPLOAD_MAX_WORKERS = 3  # 文件上传最大线程数（per uploader）
