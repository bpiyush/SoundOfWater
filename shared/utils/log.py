"""Loggers."""
import os
from os.path import dirname, realpath, abspath
from tqdm.auto import tqdm
import numpy as np


curr_filepath = abspath(__file__)
repo_path = dirname(dirname(dirname(curr_filepath)))
# repo_path = dirname(dirname(dirname(realpath(__file__))))

def tqdm_iterator(items, desc=None, bar_format=None, **kwargs):
    tqdm._instances.clear()
    iterator = tqdm(
        items,
        desc=desc,
        # bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
        **kwargs,
    )
    tqdm._instances.clear()
    
    return iterator


def print_retrieval_metrics_for_csv(metrics, scale=100):
    print_string = [
        np.round(scale * metrics["R1"], 3),
        np.round(scale * metrics["R5"], 3),
        np.round(scale * metrics["R10"], 3),
    ]
    if "MR" in metrics:
        print_string += [metrics["MR"]]
    print()
    print("Final metrics: ", ",".join([str(x) for x in print_string]))
    print()


def print_update(update, fillchar=":", color="yellow", pos="center"):
    from termcolor import colored
    # add ::: to the beginning and end of the update s.t. the total length of the
    # update spans the whole terminal
    try:
        terminal_width = os.get_terminal_size().columns - 2
    except:
        terminal_width = 98
    if pos == "center":
        update = update.center(len(update) + 2, " ")
        update = update.center(terminal_width, fillchar)
    elif pos == "left":
        update = update.ljust(terminal_width, fillchar)
        update = update.ljust(len(update) + 2, " ")
    elif pos == "right":
        update = update.rjust(terminal_width, fillchar)
        update = update.rjust(len(update) + 2, " ")
    else:
        raise ValueError("pos must be one of 'center', 'left', 'right'")
    print(colored(update, color))


if __name__ == "__main__":
    print("Repo path:", repo_path)