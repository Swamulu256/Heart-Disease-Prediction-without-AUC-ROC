import logging
import os

def setup_logging(script_name):
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)

    # 1. Create a 'logs' folder if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # 2. Use a RELATIVE path (works on Windows & Render)
    log_path = os.path.join('logs', f'{script_name}.log')

    # Create handler
    handler = logging.FileHandler(log_path, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger