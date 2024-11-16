"""Defines the audio model for pitch estimation."""
import torch
import torch.nn as nn

import math
import numpy as np
import einops

import shared.utils as su


class TimeEncodingDiscreteSinusoidal(nn.Module):
    def __init__(self, d, v=10000, rate=49, scale_factor=0.01):
        """
        Args:
            d (int): Dimension
            rate (int): discretisation rate (frames per second)
                this means that each [1/49.] of a second will be
                encoded with a unique vector
        """
        super().__init__()
        self.d = d
        self.rate = rate
        self.v = v
        self.scale_factor = scale_factor

    def forward(self, t):
        """
        Takes in timestamps t (seconds) and outputs vectors that represent these.

        Args:
            t (torch.tensor): time stamps in seconds, [B, N]
        """
        B, N = t.shape

        # Discretise time
        i = (t * self.rate).to(int)

        pe = torch.zeros(B, N, self.d).to(t.device)
        div_term = torch.exp(
            (torch.arange(0, self.d, 2, dtype=torch.float) * -(math.log(self.v) / self.d))
        )
        div_term = div_term.to(t.device)
        pe[:, :, 0::2] = torch.sin(i[:, :, None].float() * div_term)
        pe[:, :, 1::2] = torch.cos(i[:, :, None].float() * div_term)

        pe = pe * self.scale_factor

        return pe


class Wav2Vec2WithTimeEncoding(nn.Module):
    def __init__(
            self, model_name="facebook/wav2vec2-base-960h", use_time=True,
            d=512, v=10000, rate=49, scale_factor=0.01, layer_norm=False,
        ):
        super().__init__()

        su.log.print_update(
            f" [:::] Loading backbone Wav2Vec 2.0 model",
            pos="left",
            fillchar=".",
        )

        # Load pre-trained Wav2Vec 2.0 model
        from transformers import Wav2Vec2Model
        self.net = Wav2Vec2Model.from_pretrained(model_name)

        self.d = d
        self.v = v
        self.rate = rate
        self.sr = 16000
        self.use_time = use_time

        if self.use_time:
            self.time_encoding = TimeEncodingDiscreteSinusoidal(
                d=d, v=v, rate=rate, scale_factor=scale_factor,
            )
        else:
            print(" [:::] Not using time encoding.")
            self.time_encoding = None

        # Have a layer norm for the time encoding
        if layer_norm:
            self.layer_norm = nn.LayerNorm(d)
        else:
            self.layer_norm = nn.Identity()

    def forward(self, x, t):
        """
        Args:
            x (torch.tensor): audio input, [B, NC, C, NS],
                NC: n.o. clips, NS: n.o. samples
            t (torch.tensor): time stamps in seconds, [B, NC, 2],
                start and end times for each clip
        """
        B, T, C, NS = x.shape
        assert C == 1, "Require a single-channel input."
        assert t.shape[1] == T, \
            "Number of timestamps should match number of clips."
        assert t.shape[0] == B, \
            "Batch size should match."
        assert t.shape[2] == 2, \
            "Timestamps should have start and end times."

        # # Compute number of frames
        # NF = int((NS / self.sr) * self.rate)

        # Process inputs
        x = einops.rearrange(x, "B T 1 NS -> (B T) NS")
        t = einops.rearrange(t, "B T L -> (B T) L")

        # This forward is based on Huggingface's implementation of Wave2Vec2
        # https://github.com/huggingface/transformers/blob/main/src/
        # transformers/models/wav2vec2/modeling_wav2vec2.py

        # Encode through the CNN
        extract_features = self.net.feature_extractor(x)
        extract_features = extract_features.transpose(1, 2)

        if self.use_time:
            # Process timestamps: get timestamps for each frame
            # within each clip (fps=49)
            NF = extract_features.shape[1]
            t_dense = []
            for i in range(B):
                start, end = t[i]
                t_dense.append(torch.linspace(start, end, NF))
            t_dense = torch.stack(t_dense).to(extract_features.device)

            # Add time encoding to the features
            t_dense_enc = self.time_encoding(t_dense)

            # Normalise time encoding to have the same scale as the features
            extract_features = extract_features + t_dense_enc
        else:
            pass

        # Apply layer norm
        extract_features = self.layer_norm(extract_features)

        # Project into the feature space
        hidden_states, extract_features = self.net.feature_projection(
            extract_features
        )

        # Pass through the transformer encoder
        encoder_outputs = self.net.encoder(
            hidden_states,
            attention_mask=None,
            output_attentions=False, 
            output_hidden_states=False, 
            return_dict=True,
        )
        z = encoder_outputs[0]

        # z = self.backbone(x).last_hidden_state
        z = einops.rearrange(z, "(B T) F D -> B T F D", B=B, T=T)

        return z


if __name__ == "__main__":
    # Test backbone
    backbone = Wav2Vec2WithTimeEncoding()
    su.misc.num_params(backbone)

    # Test on a real audio clip
    path = "./media_assets/pouring_water_in_a_glass.wav"
    import torchaudio
    waveform, sr = torchaudio.load(path)
    waveform = torchaudio.functional.resample(waveform, sr, 16000)
    sr = 16000
    waveform = waveform.mean(dim=0, keepdim=True)

    # Forward pass an entire audio
    from transformers import Wav2Vec2Processor
    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)

    s, e = 8, 22
    x = processor(
        waveform[:, int(s*sr):int(e*sr)], sampling_rate=16000, return_tensors="pt",
    ).input_values.unsqueeze(0)
    duration = waveform.shape[-1] / sr
    t = torch.tensor([[s, e]]).unsqueeze(0)
    z = backbone(x, t)

    # Let's look at the tsne
    z_flat = einops.rearrange(z, "B T F D -> (B T F) D")
    import matplotlib.pyplot as plt
    su.visualize.show_temporal_tsne(z_flat.detach().numpy(), show=False);
    plt.savefig("./media_assets/tsne.png")
    