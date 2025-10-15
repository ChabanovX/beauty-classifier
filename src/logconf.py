import logging
import json
from datetime import datetime
import sys

from pydantic import BaseModel


class Logging(BaseModel):
    config: dict = {}  # set in Config.__post_init__
    level: str = "INFO"  # this too

    max_bytes: int
    backup_count: int
    file: str

    @property
    def prod_config(self) -> dict:
        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "file": record.filename,
                    "function": record.funcName,
                    "line": record.lineno,
                }
                if record.exc_info:
                    log_entry["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_entry)

        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"json": {"()": JsonFormatter}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.level,
                    "formatter": "json",
                    "stream": sys.stdout,
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": self.level,
                    "formatter": "json",
                    "filename": "logs/app.log",
                    "maxBytes": self.max_bytes,  # 10MB
                    "backupCount": self.backup_count,
                },
            },
            "root": {"level": self.level, "handlers": ["console", "file"]},
            "loggers": {
                "uvicorn": {
                    "level": self.level,
                    "handlers": ["console"],
                    "propagate": False,
                }
            },
        }

    @property
    def dev_config(self) -> str:
        import colorlog

        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "colored": {
                    "()": colorlog.ColoredFormatter,
                    "fmt": "%(log_color)s%(levelname)-8s%(reset)s: %(asctime)s: %(name)s: %(message)s",
                    "datefmt": "%H:%M:%S",
                    "log_colors": {
                        "DEBUG": "cyan",
                        "INFO": "green",
                        "WARNING": "yellow",
                        "ERROR": "red",
                        "CRITICAL": "red,bg_white",
                    },
                },
                "plain": {
                    "()": logging.Formatter,
                    "format": "%(levelname)-8s: %(asctime)s: %(name)s: %(message)s",
                    "datefmt": "%H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.level,
                    "formatter": "colored",
                    "stream": sys.stdout,
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": self.level,
                    "formatter": "plain",
                    "filename": "logs/app_dev.log",
                    "maxBytes": self.max_bytes,  # 10MB
                    "backupCount": self.backup_count,
                },
            },
            "root": {"level": self.level, "handlers": ["console", "file"]},
            "loggers": {
                "uvicorn": {
                    "level": self.level,
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "ml": {
                    "level": self.level,
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
            },
        }
