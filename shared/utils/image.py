"""Image operations."""
from copy import deepcopy
from PIL import Image
import numpy as np


def center_crop(im: Image):
    width, height = im.size
    new_width = width if width < height else height
    new_height = height if height < width else width 

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    
    return im


def pad_to_square(im: Image, color=(0, 0, 0)):
    im = deepcopy(im)
    width, height = im.size

    vert_pad = (max(width, height) - height) // 2
    hor_pad = (max(width, height) - width) // 2
    
    if len(im.mode) == 3:
        color = (0, 0, 0)
    elif len(im.mode) == 1:
        color = 0
    else:
        raise ValueError(f"Image mode not supported. Image has {im.mode} channels.")
    
    return add_margin(im, vert_pad, hor_pad, vert_pad, hor_pad, color=color)


def add_margin(pil_img, top, right, bottom, left, color=(0, 0, 0)):
    """Ref: https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/"""
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def resize_image(image, new_height, new_width):
    # Convert the numpy array image to PIL Image
    pil_image = Image.fromarray(image)

    # Resize the PIL Image
    resized_image = pil_image.resize((new_width, new_height))

    # Convert the resized PIL Image back to numpy array
    resized_image_np = np.array(resized_image)

    return resized_image_np


def pad_to_width(pil_image, new_width, color=(0, 0, 0)):
    """Pad the image to the specified width."""
    # Convert the numpy array image to PIL Image
    # pil_image = Image.fromarray(image)

    # Get the current width and height of the image
    width, height = pil_image.size
    assert new_width > width, f"New width {new_width} is less than the current width {width}."

    # Calculate the padding required
    hor_pad = new_width - width

    # Add padding to the image
    padded_image = add_margin(pil_image, 0, hor_pad, 0, 0, color=color)

    # Convert the padded PIL Image back to numpy array
    # padded_image_np = np.array(padded_image)

    return padded_image
