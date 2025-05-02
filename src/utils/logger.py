class LogType:
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"

class Logger:
    """
    A simple logger class that provides methods to log messages with different severity levels.
    """
    
    def __init__(self, name: str, is_debug_mode: bool = False):
        self.__name = name
        self.__is_debug_mode = is_debug_mode

    def __log(self, message: str, log_type: str = LogType.INFO):
        print(f"[{log_type}] {self.__name}: {message}\n")

    def info(self, message: str):
        self.__log(message, LogType.INFO)

    def warning(self, message: str):
        self.__log(message, LogType.WARNING)

    def error(self, message: str):
        self.__log(message, LogType.ERROR)

    def debug(self, message: str, *args):
        if self.__is_debug_mode:
            self.__log(message, LogType.DEBUG)
    
    def enable_debug_mode(self):
        if not self.__is_debug_mode:
            self.__is_debug_mode = True
            self.debug(f"Debug mode has been enabled for {self.__name}")

    def disable_debug_mode(self):
        if self.__is_debug_mode:
            self.__is_debug_mode = False
            self.debug(f"Debug mode has been disabled for {self.__name}")