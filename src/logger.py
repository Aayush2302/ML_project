import logging
import os
from datetime import datetime

# ✅ Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# ✅ Generate unique log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)

# ✅ Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] [%(levelname)s] [%(lineno)d] %(name)s - %(message)s",
    level=logging.INFO,
)

logging.info("Logging has started successfully.")

# ✅ Optional: Allow easy access to log file path if needed in other modules
def get_log_file_path():
    return LOG_FILE_PATH
