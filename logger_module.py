import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

LOG_FORMAT = "%(name)s : %(levelname)s : %(asctime)s : %(message)s"
formatter = logging.Formatter(LOG_FORMAT)

file_handler = logging.FileHandler('my_logs.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


logger.info(f"info about my my_module_name")
logger.warning(f"warning message from my my_module_name")