custom_css = """
<style>
    .container {
        max-width: 100% !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    .header {
        padding: 30px;
        margin-bottom: 30px;
        text-align: center;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .header h1 {
        font-size: 36px;
        margin-bottom: 15px;
        font-weight: bold;
        color: #333333;  /* Explicitly set heading color */
    }
    .header h2 {
        font-size: 24px;
        margin-bottom: 10px;
        color: #333333;  /* Explicitly set subheading color */
    }
    .header p {
        font-size: 18px;
        margin: 5px 0;
        color: #666666;
    }
    .blue-text {
        color: #4a90e2;
    }
    /* Custom styles for slider container */
    .slider-container {
        background-color: white !important;
        padding-top: 0.9em;
        padding-bottom: 0.9em;
    }
    /* Add gap before examples */
    .examples-holder {
        margin-top: 2em;
    }
    /* Set fixed size for example videos */
    .gradio-container .gradio-examples .gr-sample {
        width: 240px !important;
        height: 135px !important;
        object-fit: cover;
        display: inline-block;
        margin-right: 10px;
    }
    .gradio-container .gradio-examples {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    /* Ensure the parent container does not stretch */
    .gradio-container .gradio-examples {
        max-width: 100%;
        overflow: hidden;
    }
    /* Additional styles to ensure proper sizing in Safari */
    .gradio-container .gradio-examples .gr-sample img {
        width: 240px !important;
        height: 135px !important;
        object-fit: cover;
    }
</style>
"""

custom_html = custom_css + """
<div class="header">
    <h1><span class="blue-text">The Sound of Water</span>: Inferring Physical Properties from Pouring Liquids</h1>
    <p><a href='https://bpiyush.github.io/pouring-water-website/'>Project Page</a> |
    <a href='https://github.com/bpiyush/SoundOfWater'>Github</a> | 
    <a href='#'>Paper</a> |
    <a href='https://huggingface.co/datasets/bpiyush/sound-of-water'>Data</a>
    <a href='https://huggingface.co/bpiyush/sound-of-water-models'>Models</a></p>
</div>
"""

tips = """
<div>
<br><br>
Please give us a 🌟 on <a href='https://github.com/bpiyush/SoundOfWater'>Github</a> if you like our work!
Tips to get better results:
<ul>
    <li>Make sure there is not too much noise such that the pouring is audible.</li>
    <li>The video is not used during the inference.</li>
</ul>
</div>
"""

import os
import sys

import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import decord
import PIL, PIL.Image
import librosa
from IPython.display import Markdown, display
import pandas as pd

from demo.util import (
	custom_css,
	custom_html,
	read_html_file,
)
import shared.utils as su
import sound_of_water.audio_pitch.model as audio_models
import sound_of_water.data.audio_loader as audio_loader
import sound_of_water.data.audio_transforms as at
import sound_of_water.data.csv_loader as csv_loader


def read_html_file(file):
    with open(file) as f:
        return f.read()



def define_axes(figsize=(13, 4), width_ratios=[0.22, 0.78]):
    fig, axes = plt.subplots(
        1, 2, figsize=figsize, width_ratios=width_ratios,
        layout="constrained",
    )
    return fig, axes


def show_frame_and_spectrogram(frame, spectrogram, visualise_args, axes=None):
    """Shows the frame and spectrogram side by side."""

    if axes is None:
        fig, axes = define_axes()
    else:
        assert len(axes) == 2

    ax = axes[0]
    ax.imshow(frame, aspect="auto")
    ax.set_title("Example frame")
    ax.set_xticks([])
    ax.set_yticks([])
    ax = axes[1]
    audio_loader.show_logmelspectrogram(
        S=spectrogram,
        ax=ax,
        show=False,
        sr=visualise_args["sr"],
        n_fft=visualise_args["n_fft"],
        hop_length=visualise_args["hop_length"],
    )


def scatter_pitch(ax, t, f, s=60, marker="o", color="limegreen", label="Pitch"):
    """Scatter plot of pitch."""
    ax.scatter(t, f, color=color, label=label, s=s, marker=marker)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.legend(loc="upper left")


# Load video frame
def load_frame(video_path):
    vr = decord.VideoReader(video_path, num_threads=1)
    frame = PIL.Image.fromarray(vr[0].asnumpy())
    frame = audio_loader.crop_or_pad_to_size(frame, size=(270, 480))
    return frame


def load_spectrogram(video_path):
    y = audio_loader.load_audio_clips(
        audio_path=video_path,
        clips=None,
        load_entire=True,
        cut_to_clip_len=False,
        **aload_args,
    )[0]
    S = audio_loader.librosa_harmonic_spectrogram_db(
        y,
        sr=visualise_args["sr"],
        n_fft=visualise_args["n_fft"],
        hop_length=visualise_args["hop_length"],
        n_mels=visualise_args['n_mels'],
    )
    return S


# Load audio
visualise_args = {
    "sr": 16000,
    "n_fft": 400,
    "hop_length": 320,
    "n_mels": 64,
    "margin": 16.,
    "C": 340 * 100.,
    "audio_output_fps": 49.,
    "w_max": 100.,
    "n_bins": 64,
}
aload_args = {
    "sr": 16000,
    "clip_len": None,
    "backend": "decord",
}


cfg_backbone = {
    "name": "Wav2Vec2WithTimeEncoding",
    "args": dict(),
}
backbone = getattr(audio_models, cfg_backbone["name"])(
    **cfg_backbone["args"],
)


cfg_model = {
    "name": "WavelengthWithTime",
    "args": {
        "axial": True,
        "axial_bins": 64,
        "radial": True,
        "radial_bins": 64,
        "freeze_backbone": True,
        "train_backbone_modules": [6, 7, 8, 9, 10, 11],
        "act": "softmax",
        "criterion": "kl_div",
    }
}


def load_model():
    model = getattr(audio_models, cfg_model["name"])(
        backbone=backbone, **cfg_model["args"],
    )
    su.misc.num_params(model)


    # Load the model weights from trained checkpoint
    # NOTE: Be sure to set the correct path to the checkpoint
    su.log.print_update("[:::] Loading checkpoint ", color="cyan", fillchar=".")
    ckpt_dir = "/work/piyush/pretrained_checkpoints/SoundOfWater"
    ckpt_path = os.path.join(
        ckpt_dir, 
        "dsr9mf13_ep100_step12423_real_finetuned_with_cosupervision.pth",
    )
    assert os.path.exists(ckpt_path), \
        f"Checkpoint not found at {ckpt_path}."
    print("Loading checkpoint from: ", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    msg = model.load_state_dict(ckpt)
    print(msg)
    return model


# Define audio transforms
cfg_transform = {
    "audio": {
        "wave": [
            {
                "name": "AddNoise",
                "args": {
                "noise_level": 0.001
                },
                "augmentation": True,
            },
            {
                "name": "ChangeVolume",
                "args": {
                "volume_factor": [0.8, 1.2]
                },
                "augmentation": True,
            },
            {
                "name": "Wav2Vec2WaveformProcessor",
                "args": {
                "model_name": "facebook/wav2vec2-base-960h",
                "sr": 16000
                }
            }
        ],
        "spec": None,
    }
}
audio_transform = at.define_audio_transforms(
    cfg_transform, augment=False,
)

# Define audio pipeline arguments
apipe_args = {
    "spec_args": None,
    "stack": True,
}


def load_audio_tensor(video_path):
    # Load and transform input audio
    audio = audio_loader.load_and_process_audio(
        audio_path=video_path,
        clips=None,
        load_entire=True,
        cut_to_clip_len=False,
        audio_transform=audio_transform,
        aload_args=aload_args,
        apipe_args=apipe_args,
    )[0]
    return audio


def get_model_output(audio, model):
    with torch.no_grad():
        NS = audio.shape[-1]
        duration = NS / 16000
        t = torch.tensor([[0, duration]]).unsqueeze(0)
        x = audio.unsqueeze(0)
        z_audio = model.backbone(x, t)[0][0].cpu()
        y_audio = model(x, t)["axial"][0][0].cpu()
    return z_audio, y_audio


def show_output(frame, S, y_audio, z_audio):
    # duration = S.shape[-1] / visualise_args["sr"]
    # print(S.shape, y_audio.shape, z_audio.shape)
    duration = librosa.get_duration(
        S=S,
        sr=visualise_args["sr"],
        n_fft=visualise_args["n_fft"],
        hop_length=visualise_args["hop_length"],
    )
    timestamps = np.linspace(0., duration, 25)

    # Get timestamps at evaluation frames
    n_frames = len(y_audio)
    timestamps_eval = librosa.frames_to_time(
        np.arange(n_frames),
        sr=visualise_args['sr'],
        n_fft=visualise_args['n_fft'],
        hop_length=visualise_args['hop_length'],
    )
    # Get predicted frequencies at these times
    wavelengths = y_audio @ torch.linspace(
        0, visualise_args['w_max'], visualise_args['n_bins'],
    )
    f_pred = visualise_args['C'] / wavelengths
    # Pick only those timestamps where we define the true pitch
    indices = su.misc.find_nearest_indices(timestamps_eval, timestamps)
    f_pred = f_pred[indices]

    # print(timestamps, f_pred)

    # Show the true/pref pitch overlaid on the spectrogram
    fig, axes = define_axes()
    show_frame_and_spectrogram(frame, S, visualise_args, axes=axes)
    scatter_pitch(axes[1], timestamps, f_pred, color="white", label="Estimated pitch", marker="o", s=70)
    axes[1].set_title("True and predicted pitch overlaid on the spectrogram")
    # plt.show()
    # Convert to PIL Image and return the Image
    from PIL import Image

    # Draw the figure to a canvas
    canvas = fig.canvas
    canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = canvas.tostring_rgb()

    # Create a PIL image from the RGB data
    image = Image.frombytes("RGB", (w, h), buf)


    # Get physical properties
    l_pred = su.physics.estimate_length_of_air_column(wavelengths)
    l_pred_mean = l_pred.mean().item()
    l_pred_mean = np.round(l_pred_mean, 2)
    H_pred = su.physics.estimate_cylinder_height(wavelengths)
    H_pred = np.round(H_pred, 2)
    R_pred = su.physics.estimate_cylinder_radius(wavelengths)
    R_pred = np.round(R_pred, 2)
    # print(f"Estimated length: {l_pred_mean} cm, Estimated height: {H_pred} cm, Estimated radius: {R_pred} cm")
    df_show = pd.DataFrame({
        "Physical Property": ["Container height", "Container radius", "Length of air column (mean)"],
        "Estimated Value (in cms)": [H_pred, R_pred, l_pred_mean],
    })


    tsne_image = su.visualize.show_temporal_tsne(
        z_audio.detach().numpy(), timestamps_eval, show=False,
        figsize=(6, 5), title="Temporal t-SNE of latent features",
        return_as_pil = True,
    )

    return image, df_show, tsne_image
