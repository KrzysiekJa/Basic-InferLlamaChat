import sys

from loguru import logger


def init_logging():
    logger.remove()
    logger.add(sys.stdout, level="DEBUG", colorize=True, backtrace=True, diagnose=True)
    logger.add(
        "logs/app.log",
        rotation="30 MB",
        retention="14 days",
        serialize=True, # Store logs in JSON format for better structure and parsing
        enqueue=True,
        compression="zip",
        level="WARNING",
        backtrace=True,
        diagnose=True,
        format="{time:MMMM D, YYYY > HH:mm:ss!UTC} | {level} | {message} | {extra}",
    )
    logger.info("Logging has been set up successfully.")
