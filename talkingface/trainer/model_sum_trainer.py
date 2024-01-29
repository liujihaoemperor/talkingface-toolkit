import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
from omegaconf import OmegaConf

from utils.utils import get_commit_hash
from model.text_to_speech_talkingface.cotatron_model import Cotatron
from utils.loggers import TacotronLogger
from model.voice_convertion_talkingface.synthesizer_model import Synthesizer
from utils.loggers import SynthesizerLogger
from model.voice_convertion_talkingface.gta_extractor_model import GtaExtractor

from torch.utils.data import DataLoader
import shutil
import pytorch_lightning as pl

from model.voice_convertion_talkingface.synthesizer_model import Synthesizer
from data.text import Language
from data.dataset.text2mel_dataset import TextMelDataset, text_mel_collate

from logging import getLogger
from time import time
import dlib, json, subprocess
import torch.nn.functional as F
import glob
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
import torch.cuda.amp as amp
from torch import nn
from pathlib import Path
import math
from scipy.special import gamma
import random
import torch.utils.data

from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

from talkingface.utils import(
    ensure_dir,
    get_local_time,
    early_stopping,
    calculate_valid_score,
    dict2str,
    get_tensorboard,
    set_color,
    get_gpu_usage,
    WandbLogger
)
from talkingface.data.dataprocess.wav2lip_process import Wav2LipAudio
from talkingface.evaluator import Evaluator
    
MAX_WAV_VALUE = 32768.0

class exTrainer(Trainer):
    def __init__(self, config, model):
        super(Wav2LipTrainer, self).__init__(config, model)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            the averaged loss of this epoch
        """
        self.model.train()



        loss_func = loss_func or self.model.calculate_loss
        total_loss_dict = {}
        step = 0
        iter_data = (
            tqdm(
            train_data,
            total=len(train_data),
            ncols=None,
            )
            if show_progress
            else train_data
        )

        for batch_idx, interaction in enumerate(iter_data):
            self.optimizer.zero_grad()
            step += 1
            losses_dict = loss_func(interaction)
            loss = losses_dict["loss"]

            for key, value in losses_dict.items():
                if key in total_loss_dict:
                    if not torch.is_tensor(value):
                        total_loss_dict[key] += value
                    else:
                        losses_dict[key] = value.item()
                        total_loss_dict[key] += value.item()
                else:
                    if not torch.is_tensor(value):
                        total_loss_dict[key] = value
                    else:
                        losses_dict[key] = value.item()
                        total_loss_dict[key] = value.item()
            iter_data.set_description(set_color(f"train {epoch_idx} {losses_dict}", "pink"))

            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
        average_loss_dict = {}
        for key, value in total_loss_dict.items():
            average_loss_dict[key] = value/step

        return average_loss_dict

    
    def _valid_epoch(self, valid_data, loss_func=None, show_progress=False):
        print('Valid'.format(self.eval_step))
        self.model.eval()
        total_loss_dict = {}
        iter_data = (
            tqdm(valid_data,
                total=len(valid_data),
                ncols=None,
                desc=set_color("Valid", "pink")
            )
            if show_progress
            else valid_data
        )
        step = 0
        for batch_idx, batched_data in enumerate(iter_data):
            step += 1
            losses_dict = self.model.calculate_loss(batched_data, valid=True)
            for key, value in losses_dict.items():
                if key in total_loss_dict:
                    if not torch.is_tensor(value):
                        total_loss_dict[key] += value
                    else:
                        losses_dict[key] = value.item()
                        total_loss_dict[key] += value.item()
                else:
                    if not torch.is_tensor(value):
                        total_loss_dict[key] = value
                    else:
                        losses_dict[key] = value.item()
                        total_loss_dict[key] = value.item()
        average_loss_dict = {}
        for key, value in total_loss_dict.items():
            average_loss_dict[key] = value/step
        if losses_dict["sync_loss"] < .75:
            self.model.config["syncnet_wt"] = 0.01
        return average_loss_dict
    
    
def cota_train(config=True, gpus=None, name=True, ckpt=None, sav_topk=-1, deb=False, epoch=1):
    args = config, gpus, name, ckpt, sav_topk, deb, epoch
    model = Cotatron(args)

    hp_global = OmegaConf.load(args.config[0])
    hp_cota = OmegaConf.load(args.config[1])

    hp = OmegaConf.merge(hp_global, hp_cota)

    save_path = os.path.join(hp.log.chkpt_dir, args.name)
    os.makedirs(save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(hp.log.chkpt_dir, args.name),
        monitor='val_loss',
        verbose=True,
        save_top_k=args.save_top_k, # save all
        prefix=get_commit_hash(),
    )

    tb_logger = TacotronLogger(
        save_dir=hp.log.log_dir,
        name=args.name,
    )

    trainer = Trainer(
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        default_root_dir=save_path,
        gpus=-1 if args.gpus is None else args.gpus,
        accelerator=None,
        num_sanity_val_steps=1,
        resume_from_checkpoint=args.checkpoint_path,
        gradient_clip_val=hp.train.grad_clip,
        fast_dev_run=args.fast_dev_run,
        check_val_every_n_epoch=args.val_epoch,
        progress_bar_refresh_rate=1,
        max_epochs=10000,
    )
    trainer.fit(model)


def syn_train(config=True, gpus=None, name=True, ckpt=None, sav_topk=-1, deb=False, epoch=1):
    args = config, gpus, name, ckpt, sav_topk, deb, epoch
    model = Synthesizer(args)

    hp_global = OmegaConf.load(args.config[0])
    hp_vc = OmegaConf.load(args.config[1])

    hp = OmegaConf.merge(hp_global, hp_vc)

    save_path = os.path.join(hp.log.chkpt_dir, args.name)
    os.makedirs(save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(hp.log.chkpt_dir, args.name),
        monitor='val_loss',
        verbose=True,
        save_top_k=args.save_top_k, # save all
        prefix=get_commit_hash(),
    )

    tb_logger = SynthesizerLogger(
        save_dir=hp.log.log_dir,
        name=args.name,
    )

    if args.checkpoint_path is None:
        assert hp.train.cotatron_path is not None, \
            "pretrained aligner must be given as h.p. when not resuming"
        model.load_cotatron(hp.train.cotatron_path)

    trainer = Trainer(
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        default_root_dir=save_path,
        gpus=-1 if args.gpus is None else args.gpus,
        accelerator=None,
        num_sanity_val_steps=1,
        resume_from_checkpoint=args.checkpoint_path,
        gradient_clip_val=0.0,
        fast_dev_run=args.fast_dev_run,
        check_val_every_n_epoch=args.val_epoch,
        progress_bar_refresh_rate=1,
        max_epochs=10000,
    )
    trainer.fit(model)


def gta_extract(config, ckpt=None, length=33)
    args = config, ckpt, length
    extractor = GtaExtractor(args)
    extractor.main()


class GuidedAttentionLoss(torch.nn.Module):
    def __init__(self, guided_att_steps, guided_att_variance, guided_att_gamma):
        super(GuidedAttentionLoss, self).__init__()
        self._guided_att_steps = guided_att_steps
        self._guided_att_variance = guided_att_variance
        self._guided_att_gamma = guided_att_gamma

    def set_guided_att_steps(self, guided_att_steps):
        self._guided_att_steps = guided_att_steps

    def set_guided_att_variance(self, guided_att_variance):
        self._guided_att_variance = guided_att_variance

    def set_guided_att_gamma(self, guided_att_gamma):
        self._guided_att_gamma = guided_att_gamma

    def forward(self, alignments, input_lengths, target_lengths, global_step):
        if self._guided_att_steps < global_step:
            return 0

        self._guided_att_variance = self._guided_att_gamma ** global_step

        # compute guided attention weights (diagonal matrix with zeros on a 'blurry' diagonal)
        weights = self._compute_guided_attention_weights(
            alignments, input_lengths, target_lengths)

        # apply weights and compute mean loss
        loss = torch.sum(weights * alignments) / target_lengths.float().sum()

        return loss

    def _compute_guided_attention_weights(self, alignments, input_lengths, target_lengths):
        weights = torch.zeros_like(alignments)
        for i, (f, l) in enumerate(zip(target_lengths, input_lengths)):
            grid_f, grid_l = torch.meshgrid(
                torch.arange(f, dtype=torch.float, device=f.device),
                torch.arange(l, dtype=torch.float, device=l.device))
            weights[i, :f, :l] = 1 - torch.exp(
                -((grid_l / l - grid_f / f) ** 2) / (2 * self._guided_att_variance ** 2))
        return weights


class StaticFilter(nn.Module):
    def __init__(self, channels, kernel_size, out_dim):
        super().__init__()
        assert kernel_size % 2 == 1, \
            'kernel size of StaticFilter must be odd, got %d' % kernel_size
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            1, channels, kernel_size=kernel_size, padding=padding)
        self.fc = nn.Linear(channels, out_dim, bias=False)

    def forward(self, prev_attn):
        # prev_attn: [B, T]
        x = prev_attn.unsqueeze(1)  # [B, 1, T]
        x = self.conv(x)  # [B, channels, T]
        x = x.transpose(1, 2)  # [B, T, out_dim]
        x = self.fc(x)
        return x


class DynamicFilter(nn.Module):
    def __init__(self, channels, kernel_size, attn_rnn_dim, hypernet_dim, out_dim):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, \
            'kernel size of DynamicFilter must be odd, god %d' % kernel_size
        self.padding = (kernel_size - 1) // 2

        self.hypernet = nn.Sequential(
            nn.Linear(attn_rnn_dim, hypernet_dim),
            nn.Tanh(),
            nn.Linear(hypernet_dim, channels*kernel_size),
        )
        self.fc = nn.Linear(channels, out_dim)

    def forward(self, query, prev_attn):
        # query: [B, attn_rnn_dim]
        # prev_attn: [B, T]
        B, T = prev_attn.shape
        convweight = self.hypernet(query)  # [B, channels * kernel_size]
        convweight = convweight.view(B, self.channels, self.kernel_size)
        convweight = convweight.view(B * self.channels, 1, self.kernel_size)
        prev_attn = prev_attn.unsqueeze(0)
        x = F.conv1d(prev_attn, convweight, padding=self.padding, groups=B)
        x = x.view(B, self.channels, T)
        x = x.transpose(1, 2)  # [B, T, channels]
        x = self.fc(x)  # [B, T, out_dim]
        return x


class PriorFilter(nn.Module):
    def __init__(self, causal_n, alpha, beta):
        super().__init__()
        self.causal_n = causal_n
        self.alpha = alpha
        self.beta = beta

        def beta_func(x, y):
            return gamma(x) * gamma(y) / gamma(x+y)

        def p(n, k, alpha, beta):
            def nCr(n, r):
                f = math.factorial
                return f(n) / (f(r) * f(n-r))
            return nCr(n, k) * beta_func(k+alpha, n-k+beta) / beta_func(alpha, beta)

        self.prior = np.array([
            p(self.causal_n-1, i, self.alpha, self.beta)
            for i in range(self.causal_n)[::-1]]).astype(np.float32)

        self.prior = torch.from_numpy(self.prior)
        self.prior = self.prior.view(1, 1, -1)
        self.register_buffer('prior_filter', self.prior)

    def forward(self, prev_attn):
        prev_attn = prev_attn.unsqueeze(1)
        energies = F.conv1d(
            F.pad(prev_attn, (self.causal_n-1, 0)), self.prior_filter)
        energies = energies.squeeze(1)
        energies = torch.log(torch.clamp(energies, min=1e-8))
        return energies


class Attention(nn.Module):
    def __init__(self, attn_rnn_dim, attn_dim, static_channels, static_kernel_size,
                 dynamic_channels, dynamic_kernel_size, causal_n, causal_alpha, causal_beta):
        super().__init__()
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.static_filter = StaticFilter(
            static_channels, static_kernel_size, attn_dim)
        self.dynamic_filter = DynamicFilter(
            dynamic_channels, dynamic_kernel_size, attn_rnn_dim, attn_dim, attn_dim)
        self.prior_filter = PriorFilter(causal_n, causal_alpha, causal_beta)
        self.score_mask_value = -float('inf')

    def get_alignment_energies(self, query, prev_attn):
        static_result = self.static_filter(prev_attn)
        dynamic_result = self.dynamic_filter(query, prev_attn)
        prior_result = self.prior_filter(prev_attn)

        energies = self.v(torch.tanh(
            static_result + dynamic_result)).squeeze(-1) + prior_result
        return energies

    def forward(self, attn_hidden, memory, prev_attn, mask):
        alignment = self.get_alignment_energies(attn_hidden, prev_attn)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attn_weights = F.softmax(alignment, dim=1)  # [B, T]
        context = torch.bmm(attn_weights.unsqueeze(1), memory)
        # [B, 1, T] @ [B, T, (chn.encoder + chn.speaker)] -> [B, 1, (chn.encoder + chn.speaker)]
        context = context.squeeze(1)

        return context, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, attn_rnn_dim, attn_dim, static_channels, static_kernel_size,
                 dynamic_channels, dynamic_kernel_size, causal_n, causal_alpha, causal_beta, dropout_prob):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads

        # Define multiple attention heads
        self.attention_heads = nn.ModuleList([
            Attention(attn_rnn_dim, attn_dim, static_channels, static_kernel_size,
                      dynamic_channels, dynamic_kernel_size, causal_n, causal_alpha, causal_beta)
            for _ in range(num_heads)
        ])

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, attn_hidden, memory, prev_attn, mask):
        # Apply each attention head separately
        attention_heads_output = [head(attn_hidden, memory, prev_attn, mask)[0] for head in self.attention_heads]

        # Concatenate attention head outputs along the feature dimension
        combined_output = torch.stack(attention_heads_output, dim=-1)
        combined_output = torch.mean(combined_output, dim=-1)  # Average pooling

        # Apply dropout
        combined_output = self.dropout(combined_output)

        return combined_output, None  # Return None for attn_weights for simplicity
    

class SpkClassifier(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hp.chn.speaker.token, hp.chn.speaker.token),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hp.chn.speaker.token, len(hp.data.speakers))
        )

    def forward(self, x):
        x = self.mlp(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    
class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features, condition_dim):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.projection = nn.Linear(condition_dim, 2*num_features)

    def forward(self, x, cond):
        # x: [B, num_features, T]
        # cond: [B, condition_dim]
        x = self.bn(x)
        gamma, beta = self.projection(cond).chunk(2, dim=1)
        x = gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)
        return x
    
    
class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = list()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                nn.Conv1d(channels, channels,
                          kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(0.5),
            ))
        self.cnn = nn.Sequential(*self.cnn)

        self.lstm = nn.LSTM(channels, channels//2, 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        x = self.cnn(x)  # [B, chn, T]
        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)

        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x


class SpeakerEncoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.relu = nn.ReLU()
        self.stem = nn.Conv2d(
            1, hp.chn.speaker.cnn[0], kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
        self.cnn = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=(
                3, 3), padding=(1, 1), stride=(2, 2))
            for in_channels, out_channels in zip(list(hp.chn.speaker.cnn)[:-1], hp.chn.speaker.cnn[1:])
        ])  # 80 - 40 - 20 - 10 - 5 - 3 - 2
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(channels) for channels in hp.chn.speaker.cnn
        ])
        self.gru = nn.GRU(hp.chn.speaker.cnn[-1]*2, hp.chn.speaker.token,
                          batch_first=True, bidirectional=False)

    def forward(self, x, input_lengths):
        # x: [B, mel, T]
        x = x.unsqueeze(1)  # [B, 1, mel, T]
        x = self.stem(x)
        input_lengths = (input_lengths + 1) // 2

        for cnn, bn in zip(self.cnn, self.bn):
            x = bn(x)
            x = self.relu(x)
            x = cnn(x)
            input_lengths = (input_lengths + 1) // 2

        x = x.view(x.size(0), -1, x.size(-1))  # [B, chn.speaker.cnn[-1]*2, T}]
        x = x.transpose(1, 2)  # [B, T, chn.speaker.cnn[-1]*2]

        input_lengths, indices = torch.sort(input_lengths, descending=True)
        x = torch.index_select(x, dim=0, index=indices)

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.gru.flatten_parameters()
        _, x = self.gru(x)

        x = torch.index_select(x[0], dim=0, index=torch.sort(indices)[1])
        return x

    def inference(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)

        for cnn, bn in zip(self.cnn, self.bn):
            x = bn(x)
            x = self.relu(x)
            x = cnn(x)

        x = x.view(x.size(0), -1, x.size(-1))
        x = x.transpose(1, 2)

        self.gru.flatten_parameters()
        _, x = self.gru(x)
        x = x.squeeze(1)
        return x


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)
        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class F0_Encoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.prenet_f0 = ConvNorm(
            1, hp.chn.prenet_f0,
            kernel_size=hp.ker.prenet_f0,
            padding=max(0, int(hp.ker.prenet_f0 / 2)),
            bias=True, stride=1, dilation=1)

    def forward(self, f0s):
        f0s = self.prenet_f0(f0s)
        return f0s


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + '_' + str(y.device)
                  ] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


class PreNet(nn.Module):
    def __init__(self, channels, in_dim, depth):
        super().__init__()
        sizes = [in_dim] + [channels] * depth
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size)
             for (in_size, out_size) in zip(sizes[:-1], sizes[1:])])

    # in default tacotron2 setting, we use prenet_dropout=0.5 for both train/infer.
    # you may want to set prenet_dropout=0.0 for some case.
    def forward(self, x, prenet_dropout):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=prenet_dropout, training=True)
        return x


class PostNet(nn.Module):
    def __init__(self, channels, kernel_size, n_mel_channels, depth):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.cnn = list()
        self.cnn.append(
            nn.Sequential(
                nn.Conv1d(n_mel_channels, channels,
                          kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(channels),
                nn.Tanh(),
                nn.Dropout(0.5),))

        for i in range(1, depth - 1):
            self.cnn.append(
                nn.Sequential(
                    nn.Conv1d(channels, channels,
                              kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(channels),
                    nn.Tanh(),
                    nn.Dropout(0.5),))

        self.cnn.append(
            nn.Sequential(
                nn.Conv1d(channels, n_mel_channels, kernel_size=kernel_size, padding=padding),))

        self.cnn = nn.Sequential(*self.cnn)

    def forward(self, x):
        return self.cnn(x)


class TTSDecoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.go_frame = nn.Parameter(
            torch.randn(1, hp.audio.n_mel_channels), requires_grad=True)

        self.prenet = PreNet(
            hp.chn.prenet, in_dim=hp.audio.n_mel_channels, depth=hp.depth.prenet)
        self.postnet = PostNet(
            hp.chn.postnet, hp.ker.postnet, hp.audio.n_mel_channels, hp.depth.postnet)
        self.attention_rnn = ZoneoutLSTMCell(
            hp.chn.prenet + hp.chn.encoder + hp.chn.speaker.token, hp.chn.attention_rnn, zoneout_prob=0.1)
        self.attention_layer = Attention(
            hp.chn.attention_rnn, hp.chn.attention, hp.chn.static, hp.ker.static,
            hp.chn.dynamic, hp.ker.dynamic, hp.ker.causal, hp.ker.alpha, hp.ker.beta)
        self.decoder_rnn = ZoneoutLSTMCell(
            hp.chn.attention_rnn + hp.chn.encoder + hp.chn.speaker.token, hp.chn.decoder_rnn, zoneout_prob=0.1)
        self.mel_fc = nn.Linear(
            hp.chn.decoder_rnn + hp.chn.encoder + hp.chn.speaker.token, hp.audio.n_mel_channels)

    def get_go_frame(self, memory):
        return self.go_frame.expand(memory.size(0), self.hp.audio.n_mel_channels)

    def initialize(self, memory, mask):
        B, T, _ = memory.size()
        self.memory = memory
        self.mask = mask
        device = memory.device

        attn_h = torch.zeros(B, self.hp.chn.attention_rnn).to(device)
        attn_c = torch.zeros(B, self.hp.chn.attention_rnn).to(device)
        dec_h = torch.zeros(B, self.hp.chn.decoder_rnn).to(device)
        dec_c = torch.zeros(B, self.hp.chn.decoder_rnn).to(device)

        prev_attn = torch.zeros(B, T).to(device)
        prev_attn[:, 0] = 1.0
        context = torch.zeros(B, self.hp.chn.encoder +
                              self.hp.chn.speaker.token).to(device)

        return attn_h, attn_c, dec_h, dec_c, prev_attn, context

    def decode(self, x, attn_h, attn_c, dec_h, dec_c, prev_attn, context):
        x = torch.cat((x, context), dim=-1)
        # [B, chn.prenet + (chn.encoder + chn.speaker.token)]
        attn_h, attn_c = self.attention_rnn(x, (attn_h, attn_c))
        # [B, chn.attention_rnn]

        context, prev_attn = self.attention_layer(
            attn_h, self.memory, prev_attn, self.mask)
        # context: [B, (chn.encoder + chn.speaker.token)], prev_attn: [B, T]

        x = torch.cat((attn_h, context), dim=-1)
        # [B, chn.attention_rnn + (chn.encoder + chn.speaker.token)]
        dec_h, dec_c = self.decoder_rnn(x, (dec_h, dec_c))
        # [B, chn.decoder_rnn]

        x = torch.cat((dec_h, context), dim=-1)
        # [B, chn.decoder_rnn + (chn.encoder + chn.speaker.token)]
        mel_out = self.mel_fc(x)
        # [B, audio.n_mel_channels]

        return mel_out, attn_h, attn_c, dec_h, dec_c, prev_attn, context

    def parse_decoder_outputs(self, mel_outputs, alignments):
        # 'T' is T_dec.
        mel_outputs = torch.stack(
            mel_outputs, dim=0).transpose(0, 1).contiguous()
        mel_outputs = mel_outputs.transpose(1, 2)
        # mel: [T, B, M] -> [B, T, M] -> [B, M, T]
        alignments = torch.stack(
            alignments, dim=0).transpose(0, 1).contiguous()
        # align: [T_dec, B, T_enc] -> [B, T_dec, T_enc]

        return mel_outputs, alignments

    def forward(self, x, memory, memory_lengths, output_lengths, max_input_len,
                prenet_dropout=0.5, no_mask=False, tfrate=True):
        # x: mel spectrogram for teacher-forcing. [B, M, T].
        go_frame = self.get_go_frame(memory).unsqueeze(0)
        # [B, M, T] -> [B, T, M] -> [T, B, M]
        x = x.transpose(1, 2).transpose(0, 1)
        x = torch.cat((go_frame, x), dim=0)  # [T+1, B, M]
        x = self.prenet(x, prenet_dropout)

        attn_h, attn_c, dec_h, dec_c, prev_attn, context = \
            self.initialize(memory,
                            mask=None if no_mask else ~self.get_mask_from_lengths(memory_lengths))
        mel_outputs, alignments = [], []

        decoder_input = x[0]
        while len(mel_outputs) < x.size(0) - 1:
            mel_out, attn_h, attn_c, dec_h, dec_c, prev_attn, context = \
                self.decode(decoder_input, attn_h, attn_c,
                            dec_h, dec_c, prev_attn, context)

            mel_outputs.append(mel_out)
            alignments.append(prev_attn)

            if tfrate and self.hp.train.teacher_force.rate < random.random():
                decoder_input = self.prenet(mel_out, prenet_dropout)
            else:
                decoder_input = x[len(mel_outputs)]

        mel_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, alignments)
        mel_postnet = mel_outputs + self.postnet(mel_outputs)

        # DataParallel expects equal sized inputs/outputs, hence padding
        alignments = alignments.unsqueeze(0)
        alignments = F.pad(
            alignments, (0, max_input_len[0] - alignments.size(-1)), 'constant', 0)
        alignments = alignments.squeeze(0)

        mel_outputs, mel_postnet, alignments = \
            self.mask_output(mel_outputs, mel_postnet,
                             alignments, output_lengths)
        return mel_outputs, mel_postnet, alignments

    def get_mask_from_lengths(self, lengths, max_len=None):
        if max_len is None:
            max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
        mask = (ids < lengths.unsqueeze(1))
        return mask

    def mask_output(self, mel_outputs, mel_postnet, alignments, output_lengths=None):
        if self.hp.train.mask_padding and output_lengths is not None:
            mask = ~self.get_mask_from_lengths(
                output_lengths, max_len=mel_outputs.size(-1))
            mask = mask.unsqueeze(1)  # [B, 1, T] torch.bool
            mel_outputs.masked_fill_(mask, 0.0)
            mel_postnet.masked_fill_(mask, 0.0)

        return mel_outputs, mel_postnet, alignments


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, condition_dim, dropout):
        super().__init__()
        self.cond_bn = nn.ModuleList([
            ConditionalBatchNorm1d(
                in_channels if i == 0 else out_channels, condition_dim)
            for i in range(4)])
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.cnn = nn.ModuleList([
            nn.Conv1d(in_channels if i == 0 else out_channels, out_channels,
                      kernel_size=3, dilation=2**i, padding=2**i)
            for i in range(4)])
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, z, mask=None):
        identity = x
        x = self.cnn[0](self.dropout(self.leaky_relu(self.cond_bn[0](x, z))))

        if mask is not None:
            x.masked_fill_(mask, 0.0)

        x = self.cnn[1](self.dropout(self.leaky_relu(self.cond_bn[1](x, z))))
        if mask is not None:
            x.masked_fill_(mask, 0.0)

        x = x + self.shortcut(identity)
        if mask is not None:
            x.masked_fill_(mask, 0.0)

        identity = x
        x = self.cnn[2](self.dropout(self.leaky_relu(self.cond_bn[2](x, z))))
        if mask is not None:
            x.masked_fill_(mask, 0.0)

        x = self.cnn[3](self.dropout(self.leaky_relu(self.cond_bn[3](x, z))))
        if mask is not None:
            x.masked_fill_(mask, 0.0)

        x = x + identity
        return x


class VCDecoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.stem = nn.Conv1d(hp.chn.encoder + hp.chn.residual_out,
                              hp.chn.gblock[0], kernel_size=7, padding=3)
        self.gblock = nn.ModuleList([
            GBlock(in_channels, out_channels,
                   hp.chn.speaker.token, hp.train.dropout)
            for in_channels, out_channels in
            zip(list(hp.chn.gblock)[:-1], hp.chn.gblock[1:])])
        self.final = nn.Conv1d(
            hp.chn.gblock[-1], hp.audio.n_mel_channels, kernel_size=1)

    def forward(self, x, speaker_emb, mask=None):
        # x: linguistic features + pitch info.
        # [B, chn.encoder + chn.residual_out, T_dec]
        x = self.stem(x)  # [B, chn.gblock[0], T]
        if mask is not None:
            x.masked_fill_(mask, 0.0)

        for gblock in self.gblock:
            x = gblock(x, speaker_emb, mask)
        # x: [B, chn.gblock[-1], T]

        x = self.final(x)  # [B, M, T]
        if mask is not None:
            x.masked_fill_(mask, 0.0)
        return x


class ZoneoutLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, zoneout_prob=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.zoneout_prob = zoneout_prob
        self.lstm = nn.LSTMCell(input_size, hidden_size, bias)
        self.dropout = nn.Dropout(p=zoneout_prob)

        # initialize all forget gate bias of LSTM to 1.0
        self.lstm.bias_ih[hidden_size:2*hidden_size].data.fill_(1.0)
        self.lstm.bias_hh[hidden_size:2*hidden_size].data.fill_(1.0)

    def forward(self, x, prev_hc=None):
        h, c = self.lstm(x, prev_hc)

        if prev_hc is None:
            prev_h = torch.zeros(x.size(0), self.hidden_size)
            prev_c = torch.zeros(x.size(0), self.hidden_size)
        else:
            prev_h, prev_c = prev_hc

        if self.training:
            h = (1. - self.zoneout_prob) * self.dropout(h - prev_h) + prev_h
            c = (1. - self.zoneout_prob) * self.dropout(c - prev_c) + prev_c
        else:
            h = (1. - self.zoneout_prob) * h + self.zoneout_prob * prev_h
            c = (1. - self.zoneout_prob) * c + self.zoneout_prob * prev_c

        return h, c
