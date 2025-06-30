from pathlib import Path
import logging
import sys
import json
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name
        })

# Ensure logs/ directory exists
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "app.log"

# Create logger
logger = logging.getLogger("chatbot_logger")
logger.setLevel(logging.INFO)  # Or DEBUG, if you want more verbosity

# Formatter
formatter = JSONFormatter()

# File handler
file_handler = logging.FileHandler(log_file, mode='a')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Stream handler (stdout, for Azure App Service)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

# Avoid adding duplicate handlers
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# Optional: prevent logs from propagating to root logger
logger.propagate = False
