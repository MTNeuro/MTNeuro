from contextlib import ContextDecorator
import logging


class suppress_logs(ContextDecorator):
    r"""Can be used as a context manager and as a decorator."""
    def __init__(self, level):
        self.level = getattr(logging, level.upper())

    def __enter__(self):
        self.logger = logging.getLogger()
        self.current_level = self.logger.getEffectiveLevel()
        self.logger.setLevel(self.level)
        return self

    def __exit__(self, *exc):
        self.logger.setLevel(self.current_level)
        return False
