
"""Audio loading utils."""
import os
import numpy as np
import torch
import torchaudio
import decord
import librosa
import einops
import PIL
import matplotlib.pyplot as plt
# Add serif font
plt.rcParams['font.family'] = 'serif'
from PIL import Image, ImageOps
import librosa.display

import shared.utils as su


def read_info(path):
    """
    Reads the info of the given audio file.

    Args:
        path (str): path to the audio file
    """
    import ffmpeg
    probe = ffmpeg.probe(path)
    audio_info = next(
        (s for s in probe['streams'] if s['codec_type'] == 'audio'),
        None,
    )
    video_info = next(
        (s for s in probe['streams'] if s['codec_type'] == 'video'),
        None,
    )
    return dict(video=video_info, audio=audio_info)


def load_audio_clips(
        audio_path,
        clips,
        sr,
        clip_len,
        backend='decord',
        load_entire=False,
        cut_to_clip_len=True,
    ):
    """
    Loads audio clips from the given audio file.

    Args:
        audio_path (str): path to the audio file
        clips (np.ndarray): sized [T, 2], where T is the number of clips
            and each row is a pair of start and end times of the clip
        sr (int): sample rate
        clip_len (float): length of the audio clip in seconds
        backend (str): backend to use for loading audio clips
        load_entire (bool): whether to load the entire audio file
        cut_to_clip_len (bool): whether to cut the audio clip to clip_len
    """

    if backend == 'torchaudio':
        audio_info = read_info(audio_path)["audio"]
        true_sr = int(audio_info["sample_rate"])
        true_nf = audio_info["duration_ts"]
        audio_duration = true_nf / true_sr
        # metadata = torchaudio.info(audio_path)
        # true_sr = metadata.sample_rate
        # true_nf = metadata.num_frames
    elif backend == "decord":
        # duration = librosa.get_duration(filename=audio_path)
        ar = decord.AudioReader(audio_path, sample_rate=sr, mono=True)
        # Mono=False gives NaNs in inputs.
        # This (https://gist.github.com/nateraw/fcc2bdb9c8738224957c8617c3360445) might 
        # be a related issue. Ignoring for now. Need to use torchaudio for now.
        true_nf = ar.shape[1]
        audio_duration = ar.shape[1] / sr
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if load_entire:
        # Load the entire audio as a single clip and return
        
        if backend == 'torchaudio':
            y, _ = torchaudio.load(audio_path)
            if y.shape[0] > 1:
                # Convert to a single channel
                y = y.mean(dim=0, keepdim=True)
            resampler = torchaudio.transforms.Resample(true_sr, sr)
            y = resampler(y)
            audio = y
        elif backend == "decord":
            audio = ar.get_batch(np.arange(true_nf)).asnumpy()
            audio = torch.from_numpy(audio)
        
        return [audio]

    else:
        # Clip the clips to avoid going out of bounds
        clips = np.clip(clips, 0, audio_duration)

    audio_clips = []
    for st, et in clips:

        if backend == 'torchaudio':

            # Load audio within the given time range
            sf = max(int(true_sr * st), 0)
            ef = min(int(true_sr * et), true_nf)
            nf = ef - sf
            y, _ = torchaudio.load(audio_path, frame_offset=sf, num_frames=nf)

            # Stereo to mono
            if y.shape[0] > 1:
                # Convert to a single channel
                y = y.mean(dim=0, keepdim=True)

            # Resample to the given sample rate
            resampler = torchaudio.transforms.Resample(true_sr, sr)
            y = resampler(y)

            audio = y
        
        elif backend == "decord":

            # Load audio within the given time range
            sf = max(int(st * sr), 0)
            ef = min(int(et * sr), true_nf)
            audio = ar.get_batch(np.arange(sf, ef)).asnumpy()
            audio = torch.from_numpy(audio)

            # No need to convert to mono since we are using mono=True
            # No need to resample since we are using sample_rate=sr

        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Pad the clip to clip_len
        nf_reqd = int(clip_len * sr)
        nf_curr = audio.shape[1]
        npad_side = max(0, nf_reqd - nf_curr)
        if nf_curr < nf_reqd:
            audio = torch.nn.functional.pad(audio, (0, npad_side))
        elif (nf_curr > nf_reqd) and cut_to_clip_len:
            audio = audio[:, :nf_reqd]
        
        audio_clips.append(audio)
    return audio_clips


def show_audio_clips_waveform(
        audio_clips, clips, title=None, show=True, figsize=(10, 2),
    ):
    """
    Visualizes the given audio clips.

    Args:
        audio_clips (list): list of audio clips
        sr (int): sample rate
        title (str): title of the plot
        show (bool): whether to show the clips
        figsize (tuple): figure size
    """
    clip_centers = (clips[:, 0] + clips[:, 1]) / 2
    clip_durations = clips[:, 1] - clips[:, 0]

    fig, ax = plt.subplots(1, len(audio_clips), figsize=figsize)
    if len(audio_clips) == 1:
        ax = [ax]
    for i, audio in enumerate(audio_clips):
        timestamps = np.linspace(
            clip_centers[i] - clip_durations[i],
            clip_centers[i] + clip_durations[i],
            audio.shape[-1],
        )
        ax[i].plot(timestamps, audio.squeeze().numpy(), alpha=0.5)
        ax[i].set_title(f'$t=$ {clip_centers[i]:.2f}')
        ax[i].grid(alpha=0.4)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('audio_clips_waveform.png')


# TODO: preprocess audio clips (e.g., wav-to-spectrogram, etc.)
# Note that this is different from transforms applied as augmentation 
# during training. This is more like a preprocessing step that is applied
# to the entire audio before sampling the clips.
import torchaudio.functional as TAF
import torchaudio.transforms as TAT


def load_audio(path, sr=16000, **kwargs):
    y, true_sr = torchaudio.load(path, **kwargs)
    y = y.mean(dim=0, keepdim=True)
    resampler = torchaudio.transforms.Resample(true_sr, sr)
    y = resampler(y)
    return y, sr


def load_audio_librosa(path, sr=16000, **kwargs):
    y, true_sr = librosa.load(path, sr=sr, **kwargs)
    y = torch.from_numpy(y).unsqueeze(0)
    return y, sr


def librosa_harmonic_spectrogram_db(
        y, sr=16000, n_fft=512, hop_length=256, margin=16., n_mels=64,
    ):
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    if len(y.shape) == 2:
        y = y.mean(axis=0)
    # center=True outputs 1 more frame than center=False
    # Currently, using just center=False
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=False)
    DH, DP = librosa.decompose.hpss(D, margin=margin)
    amplitude_h = np.sqrt(2) * np.abs(DH)
    if n_mels is None:
        # Usual dB spectrogram
        SH = librosa.amplitude_to_db(amplitude_h, ref=np.max)
    else:
        # Mel-scaled dB spectrogram
        S = librosa.amplitude_to_db(amplitude_h)
        SH = librosa.feature.melspectrogram(S=S, n_mels=n_mels, sr=sr)
    return SH


def show_logmelspectrogram(
        S,
        sr,
        n_fft=512,
        hop_length=256,
        figsize=(10, 3),
        ax=None,
        show=True,
        title="LogMelSpectrogram",
        xlabel="Time (s)",
        ylabel="Mel bins (Hz)",
        return_as_image=False,
    ):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    librosa.display.specshow(
        S,
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft,
        y_axis='mel',
        x_axis='time',
        ax=ax,
        auto_aspect=True,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if return_as_image:
        fig.canvas.draw()
        image = PIL.Image.frombytes(
            'RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb(),
        )
        plt.close(fig)
        return image

    if show:
        plt.show()


def show_logspectrogram(
        S, sr, n_fft=512, hop_length=256, figsize=(10, 3), ax=None, show=True,
    ):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    librosa.display.specshow(
        S,
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft,
        y_axis='linear',
        x_axis='time',
        ax=ax,
    )
    ax.set_title("LogSpectrogram")
    if show:
        plt.show()


def audio_clips_wav_to_spec(
        audio_clips, n_fft=512, hop_length=256, margin=16., n_mels=None,
    ):
    """
    Converts the given audio clips to spectrograms.

    Args:
        audio_clips (list): list of audio clips
        n_fft (int): number of FFT points
        hop_length (int): hop length
        margin (float): margin for harmonic-percussive source separation
        n_mels (int): number of mel bands (optional, if None, then dB spectrogram is returned)
    """
    audio_specs = []
    for audio in audio_clips:
        spec = librosa_harmonic_spectrogram_db(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            margin=margin,
            n_mels=n_mels,
        )
        spec = torch.from_numpy(spec).unsqueeze(0)
        audio_specs.append(spec)
    return audio_specs


def show_audio_clips_spec(
        audio_specs,
        clips,
        sr,
        n_fft=512,
        hop_length=256,
        margin=16.,
        cmap='magma',
        n_mels=None,
        show=True,
    ):
    """
    Visualizes the given audio clips.

    Args:
        audio_specs (list): list of audio spectrograms
        clips (np.ndarray): sized [T, 2], where T is the number of clips
            and each row is a pair of start and end times of the clip
        show (bool): whether to show the clips
    """
    clip_centers = (clips[:, 0] + clips[:, 1]) / 2
    clip_durations = clips[:, 1] - clips[:, 0]

    fig, ax = plt.subplots(1, len(audio_specs), figsize=(10, 4))
    if len(audio_specs) == 1:
        ax = [ax]
    for i, spec in enumerate(audio_specs):
        clip_start = clips[i][0]
        # ax[i].imshow(spec, aspect='auto', origin='lower')
        if isinstance(spec, torch.Tensor):
            spec = spec.numpy()
        if len(spec.shape) == 3:
            spec = spec[0]
        args = dict(
            data=spec,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            ax=ax[i],
            x_axis="time",
            cmap=cmap,
        )
        if n_mels is None:
            args.update(dict(y_axis="linear"))
        else:
            args.update(dict(y_axis="mel"))
        librosa.display.specshow(**args)
        # Get xticks and replace them by xticks + clip_start
        xticks = ax[i].get_xticks()
        xticks = xticks + clip_start
        ax[i].set_xticklabels([f'{x:.1f}' for x in xticks])
        ax[i].set_title(f'$t=$ {clip_centers[i]:.2f}')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('audio_clips_spec.png')


def basic_pipeline_audio_clips(
        audio_clips,
        spec_args=None,
        audio_transform=None,
        stack=True,
    ):

    wave_transform = audio_transform.get('wave', None)
    spec_transform = audio_transform.get('spec', None)

    # Apply transforms to raw waveforms
    if wave_transform is not None:
        audio_clips = wave_transform(audio_clips)

    if spec_args is not None:
        # Convert waveforms to spectrograms
        audio_clips = audio_clips_wav_to_spec(audio_clips, **spec_args)

        # Apply transforms to spectrograms
        if spec_transform is not None:
            audio_clips = spec_transform(audio_clips)

    if stack:
        audio_clips = torch.stack(audio_clips)

    return audio_clips


def load_and_process_audio(
        audio_path,
        clips,
        cut_to_clip_len=True,
        load_entire=False,
        audio_transform=None,
        aload_args=dict(),
        apipe_args=dict(),
    ):
    """Loads and preprocess audio."""

    # [C1] Load video clips: List[torch.Tensor]
    audio_clips = load_audio_clips(
        audio_path=audio_path,
        clips=clips,
        load_entire=load_entire,
        cut_to_clip_len=cut_to_clip_len,
        **aload_args,
    )

    # [C2] Pipeline:  [Preprocessing -> Transform]
    audio_clips = basic_pipeline_audio_clips(
        audio_clips=audio_clips,
        audio_transform=audio_transform,
        **apipe_args,
    )

    return audio_clips


def crop_height(image, height):
    """Crops image from the top and bottom to the desired height."""
    width, curr_height = image.size
    if curr_height < height:
        raise ValueError(f"Height of the image is less than {height}")
    top = (curr_height - height) // 2
    bottom = top + height
    return image.crop((0, top, width, bottom))


def pad_to_height(image, height):
    """Pads image with black strips at the top and bottom."""
    width, curr_height = image.size
    if curr_height > height:
        raise ValueError(f"Height of the image is already greater than {height}")
    top = (height - curr_height) // 2
    bottom = height - curr_height - top
    return ImageOps.expand(image, (0, top, 0, bottom), fill="black")


def crop_width(image, width):
    """Crops image from the left and right to the desired width."""
    curr_width, height = image.size
    if curr_width < width:
        raise ValueError(f"Width of the image is less than {width}")
    left = (curr_width - width) // 2
    right = left + width
    return image.crop((left, 0, right, height))


def crop_or_pad_height(image, height):
    """Crops or pads image to the desired height."""
    width, curr_height = image.size
    if curr_height < height:
        return pad_to_height(image, height)
    elif curr_height > height:
        return crop_height(image, height)
    return image


def crop_or_pad_width(image, width):
    """Crops or pads image to the desired width."""
    curr_width, height = image.size
    if curr_width < width:
        return pad_to_width(image, width)
    elif curr_width > width:
        return crop_width(image, width)
    return image


def pad_to_width(image, width):
    """Pads image with black strips at the left and right."""
    curr_width, height = image.size
    if curr_width > width:
        raise ValueError(f"Width of the image is already greater than {width}")
    left = (width - curr_width) // 2
    right = width - curr_width - left
    return ImageOps.expand(image, (left, 0, right, 0), fill="black")


def crop_or_pad_to_size(image, size=(270, 480)):
    """Crops or pads image to the desired size."""
    image = crop_or_pad_height(image, size[1])
    image = crop_or_pad_width(image, size[0])
    return image


if __name__ == "__main__":
    import decord
    import sound_of_water.data.audio_transforms as at

    # Testing on a sample file
    file_path = "media_assets/ayNzH0uygFw_9.0_21.0.mp4"
    assert os.path.exists(file_path), f"File not found: {file_path}"


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

    # Define audio load arguments
    aload_args = {
        "sr": 16000,
        "clip_len": None,
        "backend": "decord",
    }

    # Define audio pipeline arguments
    apipe_args = {
        "spec_args": None,
        "stack": True,
    }

    # Run the pipeline (this is used to pass to the model)
    audio = load_and_process_audio(
        audio_path=file_path,
        clips=None,
        load_entire=True,
        cut_to_clip_len=False,
        audio_transform=audio_transform,
        aload_args=aload_args,
        apipe_args=apipe_args,
    )[0]


    # This will be used to visualise
    visualise_args = {
        "sr": 16000,
        "n_fft": 400,
        "hop_length": 320,
        "n_mels": 64,
        "margin": 16.,
        "C": 340 * 100.,
        "audio_output_fps": 49.,
    }
    y = load_audio_clips(
        audio_path=file_path,
        clips=None,
        load_entire=True,
        cut_to_clip_len=False,
        **aload_args,
    )[0]
    S = librosa_harmonic_spectrogram_db(
        y,
        sr=visualise_args["sr"],
        n_fft=visualise_args["n_fft"],
        hop_length=visualise_args["hop_length"],
        n_mels=visualise_args['n_mels'],
    )

    # Load video frame
    vr = decord.VideoReader(file_path, num_threads=1)
    frame = PIL.Image.fromarray(vr[0].asnumpy())
    """
    # Cut to desired width
    new_width, new_height = 270, 480
    width, height = frame.size
    if width > new_width:
        # Crop the width
        left = (width - new_width) // 2
        right = left + new_width
        frame = frame.crop((left, 0, right, height))
    else:
        # Resize along width to have the desired width
        frame = su.visualize.resize_width(frame, new_width)
    assert frame.size[0] == new_width, \
        f"Width mismatch: {frame.size[0]} != {new_width}"

    # Now pad/crop to desired height
    if height > new_height:
        # Crop the height
        top = (height - new_height) // 2
        bottom = top + new_height
        frame = frame.crop((0, top, new_width, bottom))
    else:
        # Pad the height
        frame = pad_to_height(frame, new_height)
    assert frame.size[1] == new_height, \
        f"Height mismatch: {frame.size[1]} != {new_height}"
    """
    frame = crop_or_pad_to_size(frame)
    # frame.save("1.png")

    # Visualise
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 4), width_ratios=[0.25, 0.75],
    )
    ax = axes[0]
    ax.imshow(frame, aspect="auto")
    ax.set_title("Example frame")
    ax.set_xticks([])
    ax.set_yticks([])
    ax = axes[1]
    show_logmelspectrogram(
        S=S,
        ax=ax,
        show=False,
        sr=visualise_args["sr"],
        n_fft=visualise_args["n_fft"],
        hop_length=visualise_args["hop_length"],
    )
    plt.savefig("./media_assets/audio_visualisation.png", bbox_inches="tight")
    plt.close()
