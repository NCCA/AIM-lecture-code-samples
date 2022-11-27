import logging

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(name)s : %(levelname)s : %(asctime)s : %(message)s"
logging.basicConfig(filename="my_logs.log" ,level=logging.INFO, format=LOG_FORMAT)

logger.info(f"info about my my_module_name")
logger.warning(f"warning message from my my_module_name")
