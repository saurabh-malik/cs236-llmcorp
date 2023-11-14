import logging


def activate_logger(logger_name: str = "default",
                    format="%(asctime)s [%(module)s.%(funcName)s:%(lineno)d] %(levelname)s: %(message)s",
                    level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    formatter = logging.Formatter(format)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger