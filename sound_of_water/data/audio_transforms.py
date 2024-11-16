"""Audio transforms."""
import torchaudio
import torchvision
from torchvision.transforms import Compose, ToTensor
import torchaudio.transforms as T
import imgaug.augmenters as iaa
import numpy as np
import torch


class AddNoise(object):
    """Add noise to the waveform."""
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def __call__(self, waveform):
        noise = torch.randn_like(waveform)
        return waveform + self.noise_level * noise

    def __repr__(self):
        return self.__class__.__name__ + f"(noise_level={self.noise_level})"


class ChangeVolume(object):
    """Change the volume of the waveform."""
    def __init__(self, volume_factor=[0.6, 1.2]):
        self.volume_factor = volume_factor

    def __call__(self, waveform):
        return waveform * np.random.uniform(*self.volume_factor)

    def __repr__(self):
        return self.__class__.__name__ + f"(volume_factor={self.volume_factor})"


def configure_transforms(cfg):
    """
    Given a transform config (List[dict]), return a Compose object that
    applies the transforms in order.
    """
    transform = []
    for a in cfg:
        transform.append(eval(a["name"])(**a["args"]))
    return Compose(transform)


class AudioClipsTransform:
    def __init__(self, audio_transform):
        """Applies image transform to each frame of each video clip."""
        self.audio_transform = audio_transform

    def __call__(self, audio_clips):
        """
        Args:
            audio_clips (list): list of audio clips, each tensor [1, M]
                where M is number of samples in each clip
        """
        transformed_audio_clips = [self.audio_transform(x) for x in audio_clips]
        # transformed_audio_clips = []
        # for clip in audio_clips:
        #     transformed_clip = [self.audio_transform(x) for x in clip]
        #     transformed_audio_clips.append(transformed_clip)
        return transformed_audio_clips

    def __repr__(self):
        return self.audio_transform.__repr__()
    

class NumpyToTensor:
    def __call__(self, x):
        return torch.from_numpy(x).float()
    def __repr__(self):
        return self.__class__.__name__ + "()"


# TODO: Might have to introduce normalisation
# to have a consistent pipeline.


class Wav2Vec2WaveformProcessor:
    def __init__(self, model_name="facebook/wav2vec2-base-960h", sr=16000):
        from transformers import Wav2Vec2Processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.sr = sr
    
    def __call__(self, x):
        x = self.processor(
            x, sampling_rate=self.sr, return_tensors="pt",
        ).input_values
        return x


def define_audio_transforms(cfg_transform, augment=False):

    wave_transforms = cfg_transform["audio"]["wave"]
    wave_transforms_new = []

    # Only pick augmentations if augment=True
    for t in wave_transforms:
        if "augmentation" not in t:
            wave_transforms_new.append(t)
        else:
            if augment and t["augmentation"]:
                wave_transforms_new.append(t)
    # print(wave_transforms_new)
    wave_transform = configure_transforms(wave_transforms_new)
    wave_transform = AudioClipsTransform(wave_transform)

    # wave_transform = configure_transforms(
    #     cfg_transform["audio"]["wave"],
    # )
    # wave_transform = AudioClipsTransform(wave_transform)
    # spec_transform = configure_transforms(
    #     cfg_transform["audio"]["spec"],
    # )
    # spec_transform = AudioClipsTransform(spec_transform)

    audio_transform = dict(
        wave=wave_transform,
        # spec=spec_transform,
    )
    return audio_transform


if __name__ == "__main__":
    # Testing it out

    # Raw waveform transform
    cfg = [
        {
            "name": "AddNoise",
            "args": {"noise_level": 0.1},
        },
        {
            "name": "ChangeVolume",
            "args": {"volume_factor": [0.6, 1.2]},
        },
    ]
    transform = configure_transforms(cfg)

    x = torch.randn([1, 16000])
    z = transform(x)
    print(x.shape, z.shape)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(8, 4))
    ax[0].plot(x[0].numpy())
    ax[1].plot(z[0].numpy())
    plt.savefig("waveform_transform.png")

    # Wav2Vec2 transform
    cfg = [
        {
            "name": "Wav2Vec2WaveformProcessor",
            "args": {"model_name": "facebook/wav2vec2-base-960h", "sr": 16000},
        },
    ]
    transform = configure_transforms(cfg)
    x = torch.randn([4, 16000])
    z = transform(x)
    print(x.shape, z.shape)


    # Spectrogram transform
    cfg = [
        {
            "name": "T.FrequencyMasking",
            "args": {"freq_mask_param": 8},
        },
        {
            "name": "T.TimeMasking",
            "args": {"time_mask_param": 16},
        },
    ]
    transform = configure_transforms(cfg)
    x = torch.randn([1, 64, 251])
    z = transform(x)
    print(x.shape, z.shape)

    fig, ax = plt.subplots(2, 1, figsize=(8, 4))
    ax[0].imshow(x[0].numpy())
    ax[1].imshow(z[0].numpy())
    plt.savefig("spectrogram_transform.png")
