import matplotlib
matplotlib.use('Agg')

import logging
from rich.logging import RichHandler
logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("rich")


def set_level_all_loggers(level):
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

from rich import pretty, traceback
pretty.install(indent_guides=True)  # install in the REPL
traceback.install(indent_guides=True)  # install in the REPL

from rich.console import Console
console = Console()


from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
progress = Progress(
    TextColumn("[bold]Training", justify="right"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    "[progress.percentage]{task.completed}/{task.total:0.0f}",
    "•",
    TimeElapsedColumn(),
    "•",
    TimeRemainingColumn(),
    expand=True
)


from .prompt import new_logdir_prompt
