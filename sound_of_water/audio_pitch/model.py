"""Defines the audio model for pitch estimation."""
import torch
import torch.nn as nn
import einops

import math
import numpy as np
import einops
import pytorch_lightning as pl

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
            f" [:::] Loading backbone Wav2Vec 2.0 ",
            pos="left",
            fillchar=".",
            color="cyan",
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


def recursive_attr(module, attr):
    if "." in attr:
        m, a = attr.split(".", 1)
        return recursive_attr(getattr(module, m), a)
    return getattr(module, attr)


class WavelengthWithTime(pl.LightningModule):
    def __init__(
            self,
            backbone,
            feat_dim=768,
            axial=True,
            axial_bins=512,
            radial=True,
            radial_bins=512,
            freeze_backbone=True,
            train_backbone_modules=[10, 11], 
            prediction_head_hidden=[],
            act="softmax",
            criterion="kl_div",
            cfg_opt=dict(name="Adam", args=dict(lr=1e-4)),
        ):
        super().__init__()
        su.log.print_update(
            " [:::] Loading model WavelengthWithTime ",
            color="cyan",
            pos="left",
            fillchar=".",
        )

        # By default, freeze the entire backbone
        if freeze_backbone:
            self.freeze(backbone)
        
        # Unfreeze specific modules
        train_backbone_modules = [
            backbone.net.encoder.layers[int(m)] for m in train_backbone_modules
        ]
        for module in train_backbone_modules:
            self.unfreeze(module)
        
        # Make the layer norm in backbone trainable
        print("[>>>] Unfreezing layer norm in backbone")
        for param in backbone.layer_norm.parameters():
            param.requires_grad = True
        su.misc.num_trainable_params(backbone)

        self.backbone = backbone
        self.feat_dim = feat_dim

        # Add some intermediate layers before prediction heads
        if len(prediction_head_hidden) > 0:
            layers = []
            in_dim = feat_dim
            for out_dim in prediction_head_hidden:
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.ReLU())
                in_dim = out_dim
            self.intermediate_layers = nn.Sequential(*layers)
        else:
            self.intermediate_layers = torch.nn.Identity()
            out_dim = feat_dim
        su.misc.num_trainable_params(self.intermediate_layers)

        assert axial or radial, \
            "At least one of axial or radial heads must be enabled."

        # Define axial head
        self.axial_head = None
        if axial:
            self.axial_head = nn.Linear(out_dim, axial_bins)
            su.misc.num_trainable_params(self.axial_head)
        
        # Define radial head
        self.radial_head = None
        if radial:
            self.radial_head = nn.Linear(out_dim, radial_bins)
            su.misc.num_trainable_params(self.radial_head)

        self.act = torch.nn.Softmax(dim=-1) if act == "softmax" else torch.nn.Identity()

        # Set criterion
        self.define_criterion(criterion)

        # Define optimization config
        self.cfg_opt = cfg_opt

        # Save hyperparameters
        self.save_hyperparameters(ignore=["backbone"])

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def define_criterion(self, criterion):
        if criterion == "kl_div":
            self.criterion = nn.KLDivLoss()
        elif criterion == "ce":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Criterion {criterion} not implemented.")

    def freeze(self, net):
        for p in net.parameters():
            p.requires_grad = False

    def unfreeze(self, module):
        module_name = type(module).__name__
        print(f"[>>>] Unfreezing {module_name}")
        for p in module.parameters():
            p.requires_grad = True

    def forward(self, x, t):
        """
        Args:
            x (torch.Tensor): [B, T, C, NS], T: n.o. clips
            t (torch.Tensor): [B, T, 2], clip start and end times
        """
        B, T, C, NS = x.shape
        z = self.backbone.forward(x, t)

        # assert C == 1, "Require a single-channel input."
        # x = einops.rearrange(x, "B T 1 NS -> (B T) NS")
        
        # z = self.backbone(x).last_hidden_state
        # z = einops.rearrange(z, "(B T) F D -> B T F D", B=B, D=self.feat_dim)
        
        # Intermediate layers
        h = self.intermediate_layers(z)

        # Prediction heads
        y_pred = dict()
        if self.axial_head is not None:
            axial = self.act(self.axial_head(h))
            y_pred["axial"] = axial
        if self.radial_head is not None:
            radial = self.act(self.radial_head(h))
            y_pred["radial"] = radial
        return y_pred
    
    def compute_loss(self, y_pred: dict, y_true: dict):
        loss = dict()
        total_loss = 0.
        for key in y_pred:
            yt = y_true[key]
            yt = einops.rearrange(yt, "b t d f -> b t f d")
            yp = y_pred[key]
            if isinstance(self.criterion, nn.KLDivLoss):
                # Need to pass log to the loss function if it is KLDivLoss
                yp = yp.log()
                loss[key] = self.criterion(yp, yt)
            elif isinstance(self.criterion, nn.CrossEntropyLoss):
                yp = einops.rearrange(yp, "b t f d -> (b t f) d")
                yt = einops.rearrange(yt, "b t f d -> (b t f) d")
                loss[key] = self.criterion(yp, yt)
            else:
                raise NotImplementedError(f"Criterion {self.criterion} not implemented.")
            # For now, using hardcoded loss weights of 1/K where K is number of losses
            total_loss += loss[key] / len(y_pred)
        loss["total"] = total_loss
        return loss

    # Fill in the rest of the class definition here
    def step(self, batch, mode, log=True):
        x = batch["audio_clips"]
        t = batch["clips"]
        y_true = {**batch["targets"], **batch["metadata"]}
        y_pred = self.forward(x, t)
        losses = self.compute_loss(y_pred, y_true)
        loss = losses["total"]

        if log:
            self.log(f"batch/{mode}/loss_net", loss, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "valid")

    def configure_optimizers(self):
        function = getattr(torch.optim, self.cfg_opt["name"])
        optimizer = function(self.parameters(), **self.cfg_opt["args"])
        return optimizer


if __name__ == "__main__":
    import os

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
    # Add serif
    plt.rcParams["font.family"] = "serif"

    su.visualize.show_temporal_tsne(z_flat.detach().numpy(), show=False)
    plt.savefig("./media_assets/tsne.png")
    plt.close()


    # Test model
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
    model = eval(cfg_model["name"])(backbone=backbone, **cfg_model["args"])
    su.misc.num_trainable_params(model)

    # Load pre-trained checkpoint
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

    # Check forward pass
    x_random = torch.randn(2, 1, 1, 16000)
    t_random = torch.tensor([[[0, 1]], [[2, 3]]])
    y_pred = model(x_random, t_random)
    print("Input: ", x_random.shape)
    for key in y_pred:
        print(key, y_pred[key].shape)
    

    # Plot features with the trained backbone and save as tsne_trained.png
    z = model.backbone(x, t)
    z_flat = einops.rearrange(z, "B T F D -> (B T F) D")
    su.visualize.show_temporal_tsne(z_flat.detach().numpy(), show=False)
    plt.savefig("./media_assets/tsne_trained.png")
    plt.close()