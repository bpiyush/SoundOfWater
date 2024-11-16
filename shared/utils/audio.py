"""Audio utils"""
import librosa
import numpy as np
import matplotlib.pyplot as plt


def load_audio(audio_path: str, sr: int = None, max_duration: int = 10., start: int = 0, stop: int = None):
    """Loads audio and pads/trims it to max_duration"""
    data, sr = librosa.load(audio_path, sr=sr)
    
    if stop is not None:
        start = int(start * sr)
        stop = int(stop * sr)
        data = data[start:stop]
    
    # Convert to mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    n_frames = int(max_duration * sr)
    if len(data) > n_frames:
        data = data[:n_frames]
    elif len(data) < n_frames:
        data = np.pad(data, (0, n_frames - len(data)), "constant")
    return data, sr
    

# def compute_spectrogram(data: np.ndarray, sr: int):
#     D = librosa.stft(data)  # STFT of y
#     S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
#     return S_db


def compute_spec_freq_mean(S_db: np.ndarray, eps=1e-5):
    # Compute mean of spectrogram over frequency axis
    S_db_normalized = (S_db - S_db.mean(axis=1)[:, None]) / (S_db.std(axis=1)[:, None] + eps)
    S_db_over_time = S_db_normalized.sum(axis=0)
    return S_db_over_time


def process_audiofile(audio_path, functions=["load_audio", "compute_spectrogram", "compute_spec_freq_mean"]):
    """Processes audio file with a list of functions"""
    data, sr = load_audio(audio_path)
    for function in functions:
        if function == "load_audio":
            pass
        elif function == "compute_spectrogram":
            data = compute_spectrogram(data, sr)
        elif function == "compute_spec_freq_mean":
            data = compute_spec_freq_mean(data)
        else:
            raise ValueError(f"Unknown function {function}")
    return data



"""PyDub's silence detection is based on the energy of the audio signal."""
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SilenceDetector:
    

    def __init__(self, silence_thresh=-36) -> None:
        self.silence_thresh = silence_thresh
    
    def __call__(self, audio_path: str, start=None, end=None):
        
        import pydub
        from pydub.utils import db_to_float
        
        try:
            waveform = pydub.AudioSegment.from_file(audio_path)
        except:
            print("Error loading audio file: ", audio_path)
            return 100.0
        
        start_ms = int(start * 1000) if start else 0
        end_ms = int(end * 1000) if end else len(waveform)
        waveform = waveform[start_ms:end_ms]
        
        # convert silence threshold to a float value (so we can compare it to rms)
        silence_thresh = db_to_float(self.silence_thresh) * waveform.max_possible_amplitude
        
        if waveform.rms == 0:
            return 100.0

        silence_prob = sigmoid((silence_thresh - waveform.rms) / waveform.rms)

        # return waveform.rms <= silence_thresh
        return np.round(100 * silence_prob, 2)


def frequency_bin_to_value(bin_index, sr, n_fft):
    return int(bin_index * sr / n_fft)


def time_bin_to_value(bin_index, hop_length, sr):
    return (bin_index) * (hop_length / sr)


def add_time_annotations(ax, nt_bins, hop_length, sr, skip=50):
    # Show time (s) values on the x-axis
    t_bins = np.arange(nt_bins)
    t_vals = np.round(np.array([time_bin_to_value(tb, hop_length, sr) for tb in t_bins]), 1)
    try:
        ax.set_xticks(t_bins[::skip], t_vals[::skip])
    except:
        pass
    ax.set_xlabel("Time (s)")


def add_freq_annotations(ax, nf_bins, sr, n_fft, skip=50):
    f_bins = np.arange(nf_bins)
    f_vals = np.array([frequency_bin_to_value(fb, sr, n_fft) for fb in f_bins])
    try:
        ax.set_yticks(f_bins[::skip], f_vals[::skip])
    except:
        pass
    # ax.set_yticks(f_bins[::skip])
    # ax.set_yticklabels(f_vals[::skip])
    ax.set_ylabel("Frequency (Hz)")


def show_single_spectrogram(
    spec,
    sr,
    n_fft,
    hop_length,
    ax=None,
    fig=None,
    figsize=(10, 2),
    cmap="viridis",
    colorbar=True,
    show=True,
    format='%+2.0f dB',
    xlabel='Time (s)',
    ylabel="Frequency (Hz)",
    title=None,
    show_dom_freq=False,
):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    axim = ax.imshow(spec, origin="lower", cmap=cmap)

    # Show frequency (Hz) values on y-axis
    nf_bins, nt_bins = spec.shape

    if "frequency" in ylabel.lower():
        # Add frequency annotation
        add_freq_annotations(ax, nf_bins, sr, n_fft)

    # Add time annotation
    add_time_annotations(ax, nt_bins, hop_length, sr)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if colorbar:
        fig.colorbar(axim, ax=ax, orientation='vertical', fraction=0.01, format=format)

    if show_dom_freq:
        fmax = spec.argmax(axis=0)
        ax.scatter(np.arange(spec.shape[1]), fmax, color="white", s=0.2)

    if show:
        plt.show()


def compute_spectrogram(y, n_fft, hop_length, margin, n_mels=None):
    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # Run HPSS
    S, _ = librosa.decompose.hpss(D, margin=margin)

    # DB
    S = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    if n_mels is not None:
        S = librosa.feature.melspectrogram(S=S, n_mels=n_mels)

    return S


def show_spectrogram(S, sr, n_fft=512, hop_length=256, figsize=(10, 3), n_mels=None, ax=None, show=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    y_axis = "mel" if n_mels is not None else "linear"
    librosa.display.specshow(
        S,
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft,
        y_axis=y_axis,
        x_axis='time',
        ax=ax,
    )
    ax.set_title("LogSpectrogram" if n_mels is None else "LogMelSpectrogram")
    if show:
        plt.show()


def show_frame_and_spectrogram(frame, S, sr, figsize=(12, 4), show=True, axes=None, **spec_args):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [0.2, 0.8]})
    ax = axes[0]
    ax.imshow(frame)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    show_spectrogram(S=S, sr=sr, ax=ax, show=False, **spec_args)

    plt.tight_layout()

    if show:
        plt.show()