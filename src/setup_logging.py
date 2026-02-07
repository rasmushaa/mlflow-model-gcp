"""
The module sets up logging configuration for the application.
It configures logging to output messages to both the console and a log file,
with a specified format and logging level.
This should be called at the start of the application to ensure consistent logging behavior.
"""

import logging
import sys


def setup_logging(level=logging.DEBUG, logfile="app.log", suppress_external=False):
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d "
        "%(funcName)s() [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console logs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File logs (overwrite on each run)
    file_handler = logging.FileHandler(logfile, mode="w")
    file_handler.setFormatter(formatter)

    # Root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Remove old handlers to prevent duplicates on re-import
    if root.hasHandlers():
        root.handlers.clear()

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Suppress external library logs if requested
    if suppress_external:
        # Set all third-party loggers to WARNING or higher
        logging.getLogger("mlflow").setLevel(logging.WARNING)
        logging.getLogger("google").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("pandas").setLevel(logging.WARNING)
        logging.getLogger("sklearn").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("h5py").setLevel(logging.WARNING)
        logging.getLogger("tensorflow").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)
