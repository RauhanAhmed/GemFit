import logging
import os

# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  

# Define the directory for log files
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)  
LOG_FILE = os.path.join(LOG_DIR, "runningLogs.log")

# Initialize stream handler and file handler for console output
streamHandler = logging.StreamHandler()
fileHandler = logging.FileHandler(LOG_FILE)

# Set the logging level for each handler
streamHandler.setLevel(logging.INFO)  
fileHandler.setLevel(logging.DEBUG)    

# Configure the logging format for both handlers
logFormatter = logging.Formatter("[%(asctime)s: %(levelname)s: %(module)s: %(message)s]")
streamHandler.setFormatter(logFormatter)  
fileHandler.setFormatter(logFormatter)    

# Add the configured handlers to the logger
logger.addHandler(streamHandler)
logger.addHandler(fileHandler)