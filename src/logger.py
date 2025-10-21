import logging
import os
from datetime import datetime


# ===================================================
# 1️⃣ Create logs directory
# ===================================================
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# ===================================================
# 2️⃣ Create timestamped log file
# ===================================================
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# ===================================================
# 3️⃣ Create custom logger
# ===================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ===================================================
# 4️⃣ Create formatter (used for both console & file)
# ===================================================
formatter = logging.Formatter(
    "[%(asctime)s] %(name)s:%(lineno)d - %(levelname)s - %(message)s"
)

# ===================================================
# 5️⃣ File Handler (writes logs to file)
# ===================================================
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ===================================================
# 6️⃣ Console Handler (shows logs in terminal)
# ===================================================
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ===================================================
# 7️⃣ Test logging setup
# ===================================================
if __name__ == "__main__":
    logger.info("Logging has started successfully!")
    logger.warning("This is a sample warning message.")
    logger.error("This is a sample error message.")
    print(f"\n✅ Log file created at: {LOG_FILE_PATH}")
