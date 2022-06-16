from collections import defaultdict
import re
import os
import pathlib
import shutil

from rich import print
from rich.filesize import decimal
from rich.markup import escape
from rich.text import Text
from rich.tree import Tree
from rich.prompt import Confirm, Prompt


def walk_directory(directory: pathlib.Path, tree: Tree) -> None:
    """Recursively build a Tree with directory contents."""
    # Sort dirs first then by filename
    paths = sorted(
        pathlib.Path(directory).iterdir(),
        key=lambda path: (path.is_file(), path.name.lower()),
    )
    tensorboard_file_count = 0
    checkpoint_file_count = 0
    latest_checkpoint_id = 0
    latest_checkpoint = None
    for path in paths:
        # Remove hidden files
        if path.name.startswith("."):
            continue
        # Remove tensorboard event logs
        if path.name.startswith("events"):
            tensorboard_file_count += 1
            continue
        if path.suffix == ".pt":
            checkpoint_file_count += 1
            checkpoint_id = int(re.findall(r'(\d+)\.pt$', path.name)[0])
            if checkpoint_id > latest_checkpoint_id:
                latest_checkpoint_id = checkpoint_id
                latest_checkpoint = path
            continue
        if path.is_dir():
            style = "dim" if path.name.startswith("__") else ""
            branch = tree.add(
                f"[bold magenta]:open_file_folder: [link file://{path}]{escape(path.name)}",
                style=style,
                guide_style=style,
            )
            walk_directory(path, branch)
        else:
            text_filename = Text(path.name, "green")
            text_filename.highlight_regex(r"\..*$", "bold red")
            text_filename.stylize(f"link file://{path}")
            file_size = path.stat().st_size
            text_filename.append(f" ({decimal(file_size)})", "blue")
            try:
                create_time = path.stat().st_birthtime
                text_filename.append(f" ({create_time})", "blue")
            except:
                pass
            icon = defaultdict(lambda: "📄 ", py="🐍 ", cfg= "🛠 ", pt="🔥 ")[path.suffix[1:]]
            tree.add(Text(icon) + text_filename)
    if checkpoint_file_count:
        text_torch = Text("Found %d pytorch checkpoints, latest is " % tensorboard_file_count)
        text_filename = Text(latest_checkpoint.name, "green")
        tree.add(Text("🔥 ") + text_torch + text_filename)

    if tensorboard_file_count:
        text_tensorboard = Text("Found %d tensorboard logs" % tensorboard_file_count)
        tree.add(Text("📈 ") + text_tensorboard)


def get_dir_tree(directory):
    tree = Tree(
        f":open_file_folder: [link file://{directory}]{directory}",
        guide_style="bold bright_blue",
    )
    walk_directory(pathlib.Path(directory), tree)
    return tree


def new_logdir_prompt(directory):
    print('Log directory already exists')
    print(get_dir_tree(directory))
    rm_dir = Confirm.ask("Do you want to delete the folder? "
                         "[bold red blink]This will permanently remove its contents.[/bold red blink]",
                         default=False)
    if rm_dir:
        # todo only remove tensorboard
        #shutil.rmtree(directory)
        new_directory = directory
    else:
        new_directory = Prompt.ask("Enter new path. If empty, will add the next available increment.", default="")
        suffix_id = 1
        if new_directory == "":
            new_directory = directory
            while os.path.exists(new_directory):
                new_directory = directory.rstrip('/') + '_%d' % suffix_id
                suffix_id += 1
        else:
            if os.path.exists(new_directory):
                new_directory = new_logdir_prompt(new_directory)
    return new_directory
