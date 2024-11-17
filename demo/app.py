import os
import gradio as gr

from demo.util import (
	custom_css,
	custom_html,
	read_html_file,
)



css = """
<style>
    body {
        font-family: 'Arial', serif;
        margin: 0;
        padding: 0;
    }
    .header {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 5px;
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
        color: white;
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
        <div class="subtitle" style="font-size: 30px; margin-left: -30px;"><b>Inferring Physical Properties from Pouring Liquids</b></div>
        <div class="authors">
            <a href="https://bpiyush.github.io/">Piyush Bagad</a><sup>1</sup>,
            <a href="https://makarandtapaswi.github.io/">Makarand Tapaswi</a><sup>2</sup>,
            <a href="https://www.ceessnoek.info/">Cees G. M. Snoek</a><sup>3</sup>,
            <a href="https://www.robots.ox.ac.uk/~az/">Andrew Zisserman</a><sup>1</sup>,
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
<div class="header">
<div class="content" style="font-size: 16px;">
Please give us a 🌟 on <a href='https://github.com/bpiyush/SoundOfWater'>Github</a> if you like our work!
Tips to get better results:
<br><br>
<ul style="text-align: left; margin: -5px; padding: -5px">
    <li>Make sure there is not too much noise such that the pouring is audible.</li>
    <li>The video is not used during the inference.</li>
</ul>
</div>
</div>
"""

def greet(name, is_morning, temperature):
    salutation = "Good morning" if is_morning else "Good evening"
    greeting = f"{salutation} {name}. It is {temperature} degrees today"
    celsius = (temperature - 32) * 5 / 9
    return greeting, round(celsius, 2)


# Define Gradio interface
with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Default(
        primary_hue=gr.themes.colors.red,
        secondary_hue=gr.themes.colors.pink,
    ),
) as demo:
    # Add the header
    gr.HTML(header)

    gr.Interface(
        fn=greet,
        inputs=["text", "checkbox", gr.Slider(0, 100)],
        outputs=["text", "number"],
    )
        
    # Add the footer
    gr.HTML(footer)


# Launch the interface
demo.launch(allowed_paths=["../", "."], share=True)