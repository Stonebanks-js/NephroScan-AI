import os      # OS operations like file/directory handling
import sys     # System-specific parameters and functions
import logging # Logging library for debug/info/error messages

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"  # Log message format

log_dir = "logs"  # Directory to store log files
log_filepath = os.path.join(log_dir, "running_logs.log")  # Full log file path
os.makedirs(log_dir, exist_ok=True)  # Create log directory if not exists

# Configure logging: level, format, and handlers (file + console output)
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),      # Save logs to file
        logging.StreamHandler(sys.stdout)       # Show logs in terminal
    ]
)

logger = logging.getLogger(__name__)  # Logger instance for this module





# Q: What is the importance of a logger file?
# A: Records important events, errors, and execution flow for debugging and monitoring.

# Q: What happens if I don’t create a logger?
# A: No history of program execution, making debugging and issue tracking difficult.

# Q: What are the disadvantages of not creating logs?
# A: 
# - Cannot trace errors after execution.
# - Hard to reproduce bugs due to missing execution history.
# - No performance tracking or analysis.

# Q: Where can I get stuck without logs?
# A: 
# - During debugging, unable to identify failure points.
# - In production, no clues about what went wrong.

# Q: Why record time, date, and message in logs?
# A: Helps pinpoint when, where, and in which module an issue occurred.

# Q: How do logs help me travel through all files?
# A: Unified logging across modules lets you trace execution steps chronologically.

# Q: How can I access logs?
# A: 
# - From the logs/ directory (log file).
# - From the console output while the program runs.
