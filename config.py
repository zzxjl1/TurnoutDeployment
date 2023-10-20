TARGET_SAMPLE_RATE = 25  # 模型输入采样率，必须与之匹配
FILE_OUTPUT = False  # 是否输出文件
DEBUG = False  # 是否开启web server调试模式
FORCE_CPU = False  # 强制使用CPU跑模型
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
]  # 支持的故障类型
