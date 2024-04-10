import logging
import multiprocessing
import pickle
import logging
import socketserver
import struct
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler, SocketHandler
from config import (
    LOG_BACKUP_COUNT,
    LOG_FILE_PATH,
    LOG_ROTATION_MAX_SIZE,
    LOG_LEVEL,
    ENABLE_CONSOLE_LOG,
    ENABLE_FILE_LOG,
    LOG_ROTATION_POLICY,
    LOG_ROTATION_MAX_TIME_INTERVAL,
    LOG_ROTATION_MAX_TIME_UNIT,
    PORT
)

DEFAULT_TCP_LOGGING_PORT = PORT+1

log_levels = {
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.FATAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


logger = logging.getLogger("ROOT_logger")
assert LOG_LEVEL in log_levels, f"Invalid log level: {LOG_LEVEL}"
logger.setLevel(log_levels[LOG_LEVEL])

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

socket_handler = SocketHandler('localhost', DEFAULT_TCP_LOGGING_PORT)
socket_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(threadName)s - %(module)s:%(funcName)s:%(lineno)d - %(levelname)s: %(message)s"
)

console_handler.setFormatter(formatter)

if ENABLE_CONSOLE_LOG:
    logger.addHandler(console_handler)  # 将日志输出至控制台
if ENABLE_FILE_LOG:
    logger.addHandler(socket_handler)  # 将日志输出至文件



class LogRecordStreamHandler(socketserver.StreamRequestHandler):

    def handle(self):
        """
        Handle multiple requests - each expected to be a 4-byte length,
        followed by the LogRecord in pickle format. Logs the record
        according to whatever policy is configured locally.
        """
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            slen = struct.unpack('>L', chunk)[0]
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))
            obj = self.unPickle(chunk)
            record = logging.makeLogRecord(obj)
            self.handleLogRecord(record)

    def unPickle(self, data):
        return pickle.loads(data)

    def handleLogRecord(self, record):
        # if a name is specified, we use the named logger rather than the one
        # implied by the record.
        logger = self.server.target_logger
        # N.B. EVERY record gets logged. This is because Logger.handle
        # is normally called AFTER logger-level filtering. If you want
        # to do filtering, do it at the client end to save wasting
        # cycles and network bandwidth!
        logger.handle(record)
        # print(record)

class LogRecordSocketReceiver(socketserver.ThreadingTCPServer):

    def __init__(self, host='localhost',
                 port=DEFAULT_TCP_LOGGING_PORT,
                 handler=LogRecordStreamHandler):
        socketserver.ThreadingTCPServer.__init__(self, (host, port), handler)
        self.abort = 0
        self.timeout = 1
        self.logname = "tcplogger"
        self.target_logger = self.get_file_logger()

    def serve_until_stopped(self):
        import select
        abort = 0
        while not abort:
            rd, wr, ex = select.select([self.socket.fileno()],
                                       [], [],
                                       self.timeout)
            if rd:
                self.handle_request()
            abort = self.abort

    def get_file_logger(self):
        logger = logging.getLogger(self.logname)
        # fh = logging.FileHandler(filename='./server.log')
        fh_size = RotatingFileHandler(
            LOG_FILE_PATH,
            mode="a",
            maxBytes=LOG_ROTATION_MAX_SIZE * 1024 * 1024,
            backupCount=LOG_BACKUP_COUNT,
        )
        fh_time = TimedRotatingFileHandler(
            filename=LOG_FILE_PATH,
            when=LOG_ROTATION_MAX_TIME_UNIT,
            interval=LOG_ROTATION_MAX_TIME_INTERVAL,
            backupCount=LOG_BACKUP_COUNT
        )

        if LOG_ROTATION_POLICY == "TIME":
            file_handler = fh_time
        elif LOG_ROTATION_POLICY == "SIZE":
            file_handler = fh_size
        else:
            raise RuntimeError(f"Invaild logging rotation policy: {LOG_ROTATION_POLICY}")
        
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

