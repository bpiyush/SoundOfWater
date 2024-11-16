"""Misc utils."""
import os
from shared.utils.log import tqdm_iterator
import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        

def ignore_warnings(type="ignore"):
    import warnings
    warnings.filterwarnings(type)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def download_youtube_video(youtube_id, ext='mp4', resolution="360p", **kwargs):
    import pytube
    video_url = f"https://www.youtube.com/watch?v={youtube_id}"
    yt = pytube.YouTube(video_url)
    try:
        streams = yt.streams.filter(
            file_extension=ext, res=resolution, progressive=True, **kwargs,
        )
        # streams[0].download(output_path=save_dir, filename=f"{video_id}.{ext}")
        streams[0].download(output_path='/tmp', filename='sample.mp4')
    except:
        print("Failed to download video: ", video_url)
        return None
    return "/tmp/sample.mp4"


def check_audio(video_path):
    from moviepy.video.io.VideoFileClip import VideoFileClip
    try:
        return VideoFileClip(video_path).audio is not None
    except:
        return False


def check_audio_multiple(video_paths, n_jobs=8):
    """Parallelly check if videos have audio"""
    iterator = tqdm_iterator(video_paths, desc="Checking audio")
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs)(
            delayed(check_audio)(video_path) for video_path in iterator
        )


def num_trainable_params(model, round=3, verbose=True, return_count=False):
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    model_name = model.__class__.__name__
    if round is not None:
        value = np.round(n_params / 1e6, round)
        unit = "M"
    else:
        value = n_params
        unit = ""
    if verbose:
        print(f"::: Number of trainable parameters in {model_name}: {value} {unit}")
    if return_count:
        return n_params


def num_params(model, round=3):
    n_params = sum([p.numel() for p in model.parameters()])
    model_name = model.__class__.__name__
    if round is not None:
        value = np.round(n_params / 1e6, round)
        unit = "M"
    else:
        value = n_params
        unit = ""
    print(f"::: Number of total parameters in {model_name}: {value}{unit}")


def fix_seed(seed=42):
    """Fix all numpy/pytorch/random seeds."""
    import random
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def check_tensor(x):
    print(x.shape, x.min(), x.max())


def find_nearest_indices(a, b):
    """
    Finds the indices of the elements in `a` that are closest to each element in `b`.

    Args:
        a (np.ndarray): The array to search for the closest values.
        b (np.ndarray): The array of values to search for.
    
    Returns:
        np.ndarray: The indices of the closest values in `a` for each element in `b`.
    """
    # Reshape `a` and `b` to make use of broadcasting
    a = np.array(a)
    b = np.array(b)

    # Calculate the absolute difference between each element in `b` and all elements in `a`
    diff = np.abs(a - b[:, np.newaxis])

    # Find the index of the minimum value along the second axis (which corresponds to `a`)
    indices = np.argmin(diff, axis=1)

    return indices
