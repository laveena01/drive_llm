import os
import sys
import logging

class _StreamToLogger:
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self._buf = ""

    def write(self, msg: str):
        if not msg:
            return
        self._buf += msg
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip()
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        if self._buf.strip():
            self.logger.log(self.level, self._buf.strip())
        self._buf = ""

def setup_logging(run_dir: str, log_filename: str = "train.log") -> logging.Logger:
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, log_filename)

    logger = logging.getLogger("llm_driving")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # prevent duplicate handlers if called twice
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh = logging.FileHandler(log_path, mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler(sys.__stdout__)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # capture prints too
    sys.stdout = _StreamToLogger(logger, logging.INFO)
    sys.stderr = _StreamToLogger(logger, logging.ERROR)

    logger.info(f"[LOG] Logging initialized. File: {log_path}")
    return logger
