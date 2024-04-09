import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from config import (
    LOG_BACKUP_COUNT,
    LOG_FILE_PATH,
    LOG_ROTATION_MAX_SIZE,
    LOG_LEVEL,
    ENABLE_CONSOLE_LOG,
    ENABLE_FILE_LOG,
    LOG_ROTATION_POLICY,
    LOG_ROTATION_MAX_TIME_INTERVAL
)

log_levels = {
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.FATAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

assert LOG_LEVEL in log_levels, f"Invalid log level: {LOG_LEVEL}"

logger = logging.getLogger()
logger.setLevel(log_levels[LOG_LEVEL])
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# fh = logging.FileHandler(filename='./server.log')
fh_size = RotatingFileHandler(
    LOG_FILE_PATH,
    mode="a",
    maxBytes=LOG_ROTATION_MAX_SIZE * 1024 * 1024,
    backupCount=LOG_BACKUP_COUNT,
)
fh_time = TimedRotatingFileHandler(
    filename=LOG_FILE_PATH,
    when="D",
    interval=LOG_ROTATION_MAX_TIME_INTERVAL,
    backupCount=LOG_BACKUP_COUNT
)

if LOG_ROTATION_POLICY == "TIME":
    fh = fh_time
elif LOG_ROTATION_POLICY == "SIZE":
    fh = fh_size
else:
    raise RuntimeError(f"Invaild logging rotation policy: {LOG_ROTATION_POLICY}")

fh.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(threadName)s - %(module)s:%(funcName)s:%(lineno)d - %(levelname)s: %(message)s"
)

ch.setFormatter(formatter)
fh.setFormatter(formatter)
if ENABLE_CONSOLE_LOG:
    logger.addHandler(ch)  # 将日志输出至控制台
if ENABLE_FILE_LOG:
    logger.addHandler(fh)  # 将日志输出至文件
