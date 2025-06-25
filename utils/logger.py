import logging
import os
from datetime import datetime

def setup_logger(**params):
    model_name = params.get("model_name")

    log_dir=f"./logs/{model_name}"
    prefix = f"{model_name}_training_log"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{prefix}_{timestamp}.log")

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_filename, mode='w')
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(message)s")  # Clean format for console
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
