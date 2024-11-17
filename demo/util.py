custom_css = """
<style>
    body {
        background-color: #ffffff;
        color: #333333;  /* Default text color */
    }
    .container {
        max-width: 100% !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    .header {
        background-color: #f0f0f0;
        color: #333333;
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


def read_html_file(file):
    with open(file) as f:
        return f.read()
