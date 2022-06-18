import json
from urllib.request import urlopen

from rich.console import Console
from rich.columns import Columns
from rich.panel import Panel
import colorsys
import io
from time import process_time

from rich import box
from rich.color import Color
from rich.console import Console, ConsoleOptions, RenderGroup, RenderResult
from rich.markdown import Markdown
from rich.measure import Measurement
from rich.pretty import Pretty
from rich.segment import Segment
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

table = Table.grid(padding=1, pad_edge=True)
table.title = ":cat: MYOW :cat:"
table.add_column("HP Class", no_wrap=True, justify="center", style="bold red")
table.add_column("Values")

table.add_row(
        "Styles",
        "All ansi styles: [bold]bold[/], [dim]dim[/], [italic]italic[/italic], [underline]underline[/],"
        " [strike]strikethrough[/], [reverse]reverse[/], and even [blink]blink[/].",
    )


def get_content(user):
    """Extract text from user dict."""
    country = user["location"]["country"]
    name = f"{user['name']['first']} {user['name']['last']}"
    return f"[b]{name}[/b]\n[yellow]{country}"


console = Console()


users = json.loads(urlopen("https://randomuser.me/api/?results=30").read())["results"]
console.print(users, overflow="ignore", crop=False)
user_renderables = [Panel(get_content(user), expand=True) for user in users]
console.print(Columns(user_renderables))