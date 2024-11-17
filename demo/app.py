import os
import sys
sys.path.append("../")

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

from demo.util import *


css = """
<style>
    body {
        font-family: 'Arial', serif;
        margin: 0;
        padding: 0;
        color: black;
    }
    .header {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 5px;
        color: black;
    }
    .footer {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 5px;
    }
    .image {
        margin-right: 20px;
    }
    .content {
        text-align: center;
        color: black;
    }
    .title {
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .authors {
        color: #4a90e2;
        font-size: 1.05em;
        margin: 10px 0;
    }
    .affiliations {
        font-size: 1.em;
        margin-bottom: 20px;
    }
    .buttons {
        display: flex;
        justify-content: center;
        gap: 10px;
    }
    .button {
        background-color: #545758;
        text-decoration: none;
        padding: 8px 16px;
        border-radius: 5px;
        font-size: 1.05em;
    }
    .button:hover {
        background-color: #333;
    }
</style>
"""


header = css + """
<div class="header">
    <!-- <div class="image">
        <img src="./media_assets/pouring-water-logo5.png" alt="logo" width="100">
    </div> -->
    <div class="content">
        <img src="https://bpiyush.github.io/pouring-water-website/assets/pouring-water-logo5.png" alt="logo" width="80" style="margin-bottom: -50px; margin-right: 30px;">
        <div class="title" style="font-size: 44px; margin-left: -30px;">The Sound of Water</div>
        <div style="font-size: 30px; margin-left: -30px;"><b>Inferring Physical Properties from Pouring Liquids</b></div>
        <div class="authors">
            <a style="color: #92eaff; href="https://bpiyush.github.io/">Piyush Bagad</a><sup>1</sup>,
            <a style="color: #92eaff; href="https://makarandtapaswi.github.io/">Makarand Tapaswi</a><sup>2</sup>,
            <a style="color: #92eaff; href="https://www.ceessnoek.info/">Cees G. M. Snoek</a><sup>3</sup>,
            <a style="color: #92eaff; href="https://www.robots.ox.ac.uk/~az/">Andrew Zisserman</a><sup>1</sup>,
        </div>
        <div class="affiliations">
            <sup>1</sup>University of Oxford, <sup>2</sup>IIIT Hyderabad, <sup>3</sup>University of Amsterdam
        </div>
        
        <div class="buttons">
            <a href="#" style="color: #92eaff;" class="button">arXiv</a>
            <a href="https://bpiyush.github.io/pouring-water-website/" style="color: #92eaff;" class="button">🌐 Project</a>
            <a href="https://github.com/bpiyush/SoundOfWater" style="color: #92eaff;" class="button"> <img src="https://bpiyush.github.io/pouring-water-website/assets/github-logo.png" alt="logo" style="height:16px; float: left;"> &nbsp;Code</a>
            <a href="https://huggingface.co/datasets/bpiyush/sound-of-water" style="color: #92eaff;" class="button">🤗 Data</a>
            <a href="https://huggingface.co/bpiyush/sound-of-water-models" style="color: #92eaff;" class="button">🤗 Models</a>
            <a href="#" style="color: #92eaff;" class="button">🎯 Demo</a>
        </div>
    </div>
</div>
"""

footer = css + """
<div class="header" style="justify-content: left;">
<div class="content" style="font-size: 16px;">
Please give us a 🌟 on <a href='https://github.com/bpiyush/SoundOfWater'>Github</a> if you like our work!
Tips to get better results:
<br><br>
<ol style="text-align: left; font-size: 14px; margin-left: 30px">
    <li>Make sure there is not too much noise such that the pouring is audible.</li>
    <li>Note that the video is not used during the inference. Only the audio must be clear enough.</li>
</ol>
</div>
</div>
"""

# def process_input(video=None, youtube_link=None, start_time=None, end_time=None):
#     if video:
#         return f"Video file uploaded: {video.name}"
#     elif youtube_link and start_time and end_time:
#         return f"YouTube link: {youtube_link} (Start: {start_time}, End: {end_time})"
#     else:
#         return "Please upload a video or provide a YouTube link with start and end times."


def configure_input():
    gr.Markdown(
        "#### Either upload a video file or provide a YouTube link with start and end times."
    )
    video_input = gr.Video(label="Upload Video", height=480)
    youtube_link_start = gr.Textbox(label="YouTube Link (Start time)")
    youtube_link_end = gr.Textbox(label="YouTube Link (End time)")
    return [video_input, youtube_link_start, youtube_link_end]


# Example usage in a Gradio interface
def process_input(video, youtube_link_start, youtube_link_end):
    if video is not None:
        print(video)

        # Load model globally
        model = load_model()

        # The input is a video file path
        video_path = video

        # Load first frame
        frame = load_frame(video_path)

        # Load spectrogram
        S = load_spectrogram(video_path)

        # Load audio tensor
        audio = load_audio_tensor(video_path)

        # Get output
        z_audio, y_audio = get_model_output(audio, model)

        # Show image output
        image, df_show, tsne_image = show_output(frame, S, y_audio, z_audio)

        return image, df_show, gr.Markdown(note), tsne_image

    elif (youtube_link_start is not None) and (youtube_link_end is not None):
        # Using the provided YouTube link
        # Example: https://youtu.be/6-HVn8Jzzuk?t=10
        start_link = f"Processing YouTube link: {youtube_link_start}"
        end_link = f"Processing YouTube link: {youtube_link_end}"

        # Get video ID
        video_id = youtube_link_start.split("/")[-1].split("?")[0]
        assert video_id == youtube_link_end.split("/")[-1].split("?")[0], "Video IDs do not match"
        start_time = float(youtube_link_start.split("t=")[-1])
        end_time = float(youtube_link_end.split("t=")[-1])

        raise NotImplementedError("YouTube link processing is not implemented yet")
    else:
        return "No input provided"


def greet(name, is_morning, temperature):
    salutation = "Good morning" if is_morning else "Good evening"
    greeting = f"{salutation} {name}. It is {temperature} degrees today"
    celsius = (temperature - 32) * 5 / 9
    return greeting, round(celsius, 2)



note = """
**Note**: Radius (as well as height) estimation depends on accurate wavelength estimation towards the end.
Thus, it may not be accurate if the wavelength is not estimated correctly at the end.

$$
H = l(0) = \\frac{\lambda(0) - \lambda(T)}{4} \ \ \\text{and} \ \ R = \\frac{\lambda(T)}{4\\beta}
$$
"""


def configure_outputs():
    image_wide = gr.Image(label="Estimated pitch")
    dataframe = gr.DataFrame(label="Estimated physical properties")
    image_tsne = gr.Image(label="TSNE of features", width=300)
    markdown = gr.Markdown(label="Note")
    # ["image", "dataframe", "image", "markdown"]
    return [image_wide, dataframe, markdown, image_tsne]


# Configure pre-defined examples
examples = [
    ["../media_assets/example_video.mp4", None, None],
    ["../media_assets/ayNzH0uygFw_9.0_21.0.mp4", None, None],
    ["../media_assets/biDn0Gi6V8U_7.0_15.0.mp4", None, None],
    ["../media_assets/goWgiQQMugA_2.5_9.0.mp4", None, None],
    ["../media_assets/K87g4RvO-9k_254.0_259.0.mp4", None, None],
]


# Define Gradio interface
with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Default(),
) as demo:

    # Add the header
    gr.HTML(header)
    
    gr.Interface(
        fn=process_input,
        inputs=configure_input(),
        outputs=configure_outputs(),
        examples=examples,
    )
        
    # Add the footer
    gr.HTML(footer)


# Launch the interface
demo.launch(allowed_paths=["../", "."], share=True)