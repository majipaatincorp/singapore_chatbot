# app/logger.py

from pathlib import Path 
import logging

file_path = Path("logs/app.log")

if not file_path.exists():
    file_path.parent.mkdir(parents = True, exist_ok=True)
    file_path.touch()

# Create and configure the logger
logging.basicConfig(
    filename=str(file_path),           # Log file location
    level=logging.WARNING,                # Minimum level to log
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",  # Log format
    filemode="a"                       # Append mode
)

# Get a logger instance (module-level)
logger = logging.getLogger("chatbot_logger")