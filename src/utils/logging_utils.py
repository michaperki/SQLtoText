"""
Logging utilities for the application.
Provides colored, emoji-enhanced logging with Windows compatibility.
"""

import logging
import sys
import os
from datetime import datetime
import torch  # Added this import
import colorama
from colorama import Fore, Back, Style

# Initialize colorama for Windows compatibility
colorama.init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """Custom formatter adding colors and emojis to logging outputs with Windows compatibility"""

    level_colors = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Back.WHITE
    }

    message_indicators = {
        "training": "🚀",
        "epoch": "📈",
        "evaluation": "🎯",
        "model": "🤖",
        "dataset": "📊",
        "time": "⏱️",
        "loss": "📉",
        "save": "💾",
        "gpu": "🎮",
        "sql": "📝",
        "question": "❓",
        "loading": "📥",
        "config": "⚙️",
        "error": "❌",
        "success": "✅",
        "memory": "🧠",
        "checkpoint": "📍",
        "progress": "🔄"
    }

    def format(self, record):
        """
        Format the log record with colors, timestamps, and appropriate emojis.

        Args:
            record: The log record to format

        Returns:
            str: The formatted log message
        """
        try:
            color = self.level_colors.get(record.levelno, Fore.WHITE)
            timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')

            # Find the most relevant emoji for the message
            emoji = ""
            matched_indicator = None
            for indicator, symbol in self.message_indicators.items():
                if indicator.lower() in record.getMessage().lower():
                    if matched_indicator is None or len(indicator) > len(matched_indicator):
                        emoji = symbol + " "
                        matched_indicator = indicator

            message = record.getMessage()

            # Enhanced color formatting for SQL and Questions
            if "SQL:" in message:
                if "Rephrase this SQL query into a natural language question:" in message:
                    prefix, rest = message.split("Rephrase this SQL query into a natural language question:", 1)
                    message = f"{prefix}Rephrase this SQL query into a natural language question:{Back.BLUE}{Fore.WHITE}{rest}{Style.RESET_ALL}"
                else:
                    sql_parts = message.split("SQL:", 1)
                    message = f"{sql_parts[0]}SQL:{Back.BLUE}{Fore.WHITE}{sql_parts[1]}{Style.RESET_ALL}"

            elif "Generated Question:" in message:
                q_parts = message.split("Generated Question:", 1)
                message = f"{q_parts[0]}Generated Question:{Back.GREEN}{Fore.WHITE}{q_parts[1]}{Style.RESET_ALL}"

            elif "Test prompt:" in message:
                if "Rephrase this SQL query into a natural language question:" in message:
                    prefix, rest = message.split("Rephrase this SQL query into a natural language question:", 1)
                    message = f"{prefix}Rephrase this SQL query into a natural language question:{Back.CYAN}{Fore.WHITE}{rest}{Style.RESET_ALL}"
                else:
                    p_parts = message.split("Test prompt:", 1)
                    message = f"{p_parts[0]}Test prompt:{Back.CYAN}{Fore.WHITE}{p_parts[1]}{Style.RESET_ALL}"

            return f"{color}{timestamp}{Style.RESET_ALL} - {emoji}{message}"
        except Exception:
            # Fallback formatting if encoding issues occur
            return f"{timestamp} - {record.getMessage()}"

class LoggerFactory:
    """Factory class for creating and configuring loggers."""

    @staticmethod
    def create_logger(config):
        """
        Create and configure a logger with file and console handlers.

        Args:
            config: Configuration object containing logging settings

        Returns:
            logging.Logger: Configured logger instance
        """
        # Create log directory if it doesn't exist
        os.makedirs(config.logs_dir, exist_ok=True)

        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(config.logs_dir, f"training_log_{timestamp}.txt")

        # Create and configure file handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        # Create and configure console handler
        if sys.platform.startswith('win'):
            sys.stdout.reconfigure(encoding='utf-8')
            console_handler = logging.StreamHandler(sys.stdout)
        else:
            console_handler = logging.StreamHandler()

        console_formatter = ColoredFormatter()
        console_handler.setFormatter(console_formatter)

        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(config.logging_level)

        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Log initial messages
        logger.info(f"Initializing training run: {config.run_name}")
        logger.info(f"Log file created at: {log_file}")

        # Log system information
        logger.info(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
            logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        return logger
