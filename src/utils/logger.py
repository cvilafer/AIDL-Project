class LogType:
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"

class Logger:
    """
    A simple logger class that provides methods to log messages with different severity levels.

    Parameters:
        name (str): The name of the logger instance.

    Attributes:
        __name (str): The name of the logger instance.
    """
    def __init__(self, name: str):
        self.__name = name

    def __log(self, message: str, log_type: str = LogType.INFO):
        print(f"[{log_type}] {self.__name}: {message}\n")

    def info(self, message: str):
        self.__log(message, LogType.INFO)

    def warning(self, message: str):
        self.__log(message, LogType.WARNING)

    def error(self, message: str):
        self.__log(message, LogType.ERROR)

    def debug(self, message: str):
        self.__log(message, LogType.DEBUG)
