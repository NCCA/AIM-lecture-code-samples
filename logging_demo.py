import logging
import logger_module

# Config logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

LOG_FORMAT = "%(levelname)s : %(name)s : %(asctime)s : %(message)s"
formatter = logging.Formatter(LOG_FORMAT)

file_handler = logging.FileHandler('my_logs.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


# log data
logger.info(f"Some info about my program")
logger.warning(f"This is a warning message.")
# >>> logger_module : INFO : 2022-11-27 19:39:11,352 : info about my my_module_name
# >>> logger_module : WARNING : 2022-11-27 19:39:11,353 : warning message from my my_module_name
# >>> __main__ : INFO : 2022-11-27 19:39:11,353 : Some info about my program
# >>> __main__ : WARNING : 2022-11-27 19:39:11,353 : This is a warning message.