"""
Logging utility module.

Provides centralized logging configuration and logger instances.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add TRACE level (below DEBUG)
TRACE_LEVEL = logging.DEBUG - 5
logging.addLevelName(TRACE_LEVEL, "TRACE")

def trace(self, message, *args, **kws):
    """Log a message with severity 'TRACE'."""
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kws)

# Add SUCCESS level (between INFO and WARNING)
SUCCESS_LEVEL = logging.INFO + 5
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

def success(self, message, *args, **kws):
    """Log a message with severity 'SUCCESS'."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kws)

# Add trace and success methods to Logger class
logging.Logger.trace = trace
logging.Logger.success = success


class LoguruStyleFormatter(logging.Formatter):
    """Custom formatter that mimics Loguru's output style."""
    
    # ANSI color codes for terminal
    COLORS = {
        'TRACE': '\033[36m',      # Cyan
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[36m',       # Cyan
        'SUCCESS': '\033[32m',    # Green
        'WARNING': '\033[33m',    # Yellow/Orange
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    LEVEL_WIDTH = 8  # Width for level name padding
    
    def __init__(self, use_colors: bool = True):
        """
        Initialize the formatter.
        
        Args:
            use_colors: Whether to use ANSI color codes (default: True)
        """
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record in Loguru style."""
        # Format timestamp with milliseconds
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Get level name and map to Loguru-style names
        level_name = record.levelname
        if level_name == 'CRITICAL':
            level_name = 'ERROR'  # Loguru uses ERROR for critical
        
        # Pad level name to fixed width
        level_padded = f"{level_name:<{self.LEVEL_WIDTH}}"
        
        # Get module:function:line format
        module_path = f"{record.name}:{record.funcName}:{record.lineno}"
        
        # Format message
        message = record.getMessage()
        
        # Build the log line
        log_line = f"{timestamp} | {level_padded} | {module_path} - {message}"
        
        # Add colors for terminal output
        if self.use_colors and level_name in self.COLORS:
            color = self.COLORS[level_name]
            reset = self.COLORS['RESET']
            # Colorize the level name and message
            log_line = (
                f"{timestamp} | {color}{level_padded}{reset} | "
                f"{module_path} - {color}{message}{reset}"
            )
        
        return log_line


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    use_colors: bool = True
) -> None:
    """
    Configure the root logger with console and optional file handlers.
    Uses Loguru-style formatting with colored terminal output.
    
    Args:
        log_level: Logging level (default: logging.INFO)
        log_file: Optional log file name. If None, no file logging.
                  If provided without path, saves to log_dir.
        log_dir: Directory for log files (default: "logs")
        use_colors: Whether to use colors in terminal output (default: True)
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Console handler with colored formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = LoguruStyleFormatter(use_colors=use_colors)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_file is specified) - no colors in file
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # If log_file doesn't contain a path, prepend log_dir
        if '/' not in log_file and '\\' not in log_file:
            log_file_path = log_path / log_file
        else:
            log_file_path = Path(log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_formatter = LoguruStyleFormatter(use_colors=False)  # No colors in file
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str, log_level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        log_level: Optional log level override for this specific logger
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if log_level is not None:
        logger.setLevel(log_level)
    
    return logger


# Convenience function to setup logging with timestamped log file
def setup_logging_with_timestamp(
    log_level: int = logging.INFO,
    log_dir: str = "logs",
    prefix: str = "pipeline",
    use_colors: bool = True
) -> str:
    """
    Setup logging with a timestamped log file.
    Uses Loguru-style formatting with colored terminal output.
    
    Args:
        log_level: Logging level (default: logging.INFO)
        log_dir: Directory for log files (default: "logs")
        prefix: Prefix for the log file name (default: "pipeline")
        use_colors: Whether to use colors in terminal output (default: True)
        
    Returns:
        Path to the created log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{prefix}_{timestamp}.log"
    
    setup_logging(
        log_level=log_level,
        log_file=log_file,
        log_dir=log_dir,
        use_colors=use_colors
    )
    
    log_path = Path(log_dir) / log_file
    return str(log_path)

