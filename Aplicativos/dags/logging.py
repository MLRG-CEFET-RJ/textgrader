from fileinput import filename
import logging
import os

import colorlog
import datetime

LOGLEVEL = os.getenv("ATENA_LOGLEVEL", "INFO")


def init():
    log_format = (
        "%(asctime)s - "
        "%(name)s - "
        "%(funcName)s - "
        "%(levelname)s - "
        "%(message)s"
    )
    colorlog_format = f"%(log_color)s{log_format}"

    current_datetime = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

    log_file = f'{current_datetime}_log.txt'

    print(current_datetime)

    colorlog.basicConfig(format=colorlog_format, level=getattr(logging, LOGLEVEL), filename = log_file)
    logging.getLogger("azure").setLevel(logging.WARN)
