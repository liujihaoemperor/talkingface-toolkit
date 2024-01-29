import tqdm
from g2p_en import G2p
import os
import glob
from itertools import repeat
from multiprocessing import Pool, freeze_support
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from omegaconf import OmegaConf
from scipy.io.wavfile import write, read
import torch
import glob
from pysptk import sptk
from random import shuffle
import math


class G2PConverter:
    def __init__(self):
        self.g2p = G2p()

    def write_metadata(self, metadata, out_file):
        with open(out_file, 'w', encoding='utf-8') as f:
            for m in metadata:
                if m is None:
                    continue
                f.write('|'.join([str(x) for x in m]) + '\n')

    def load_metadata(self, path, split="|"):
        with open(path, 'r', encoding='utf-8') as f:
            metadata = [line.strip().split(split) for line in f]
        return metadata

    def convert(self, input_filename, output_filename):
        meta = self.load_metadata(input_filename)

        for idx, (audiopath, text, spk_id) in enumerate(tqdm.tqdm(meta)):
            phoneme = self.g2p(text)
            converted = ['{']
            for x in phoneme:
                if x == ' ':
                    converted.append('}')
                    converted.append('{')
                elif x == '-':
                    continue
                else:
                    converted.append(x)

            converted.append('}')
            phoneme = " ".join(str(x) for x in converted)
            phoneme = phoneme.replace(' }', '}').replace('{ ', '{')
            phoneme = phoneme.replace('0', '').replace('1', '').replace(
                '2', '').replace('{\'', '\'').replace('{...}', '...')
            meta[idx][1] = phoneme.replace(' {!}', '!').replace(
                ' {?}', '?').replace(' {.}', '.').replace(' {,}', ',')

        self.write_metadata(meta, output_filename)


class AudioResampler:
    def __init__(self):
        pass  # If there are any initialization steps, they can be added here

    @staticmethod
    def resample(wavdir, sr):
        newdir = wavdir.replace('.wav', '-{}k.wav'.format(sr // 1000))
        os.system(
            'ffmpeg -hide_banner -loglevel panic -y -i {} -ar {} {}'.format(wavdir, sr, newdir))
        os.remove(wavdir)

    def process_files(self, sampling_rate=22050, num_workers=32):
        freeze_support()

        input_paths = glob.glob(os.path.join(
            'datasets', '**', '*.wav'), recursive=True)

        with Pool(processes=args.num_workers) as p:
            list(tqdm.tqdm(p.starmap(self.resample, zip(input_paths,
                 repeat(args.sampling_rate))), total=len(input_paths)))


class F0Extractor:
    def __init__(self, config_path, num_workers=32, output_filename='f0s.txt'):
        self.hp = OmegaConf.load(config_path)
        self.num_workers = num_workers
        self.output_filename = output_filename

    def get_f0(self, audio, sampling_rate, frame_length, hop_length, f0_min, f0_max, harm_thresh):
        f0 = sptk.rapt(audio * 32768, sampling_rate, hop_length,
                       min=f0_min, max=f0_max, otype=2)
        f0 = np.clip(f0, 0, f0_max)
        return f0

    def process_speaker(self, filepath, sampling_rate, frame_length, hop_length, f0_min, f0_max, harm_thresh):
        audio, sr = self.load_wav_to_torch(filepath)
        assert sr == sampling_rate, 'sample mismatch: expected %d, got %d at %s' % (
            sampling_rate, sr, filepath)
        f0 = self.get_f0(audio.cpu().numpy(), sampling_rate,
                         frame_length, hop_length, f0_min, f0_max, harm_thresh)
        f0 = f0[np.nonzero(f0)]
        f0_sq = np.square(f0)
        square_over_frames = np.sum(f0_sq)
        sum_over_frames = np.sum(f0)
        n_frames = len(f0)
        return square_over_frames, sum_over_frames, n_frames

    def load_wav_to_torch(self, full_path):
        sampling_rate, data = self.read_wav_np(full_path)
        return torch.FloatTensor(data.astype(np.float32)), sampling_rate

    def read_wav_np(self, path):
        try:
            sr, wav = self.read_wav(path)
        except Exception as e:
            print(str(e) + path)
            return Exception

        if len(wav.shape) == 2:
            wav = wav[:, 0]

        if wav.dtype == np.int16:
            wav = wav / 32768.0
        elif wav.dtype == np.int32:
            wav = wav / 2147483648.0
        elif wav.dtype == np.uint8:
            wav = (wav - 128) / 128.0

        wav = wav.astype(np.float32)

        return sr, wav

    def read_wav(self, path):
        try:
            sr, wav = read(path)
        except Exception as e:
            print(str(e) + path)
            return Exception

        return sr, wav

    def write_metadata(self, metadata, out_file):
        with open(out_file, 'w', encoding='utf-8') as f:
            for m in metadata:
                if m is None:
                    continue
                f.write('|'.join([str(x) for x in m]) + '\n')

    def extract(self):
        with open(os.path.join(self.hp.data.train_dir, self.hp.data.train_meta), 'r', encoding='utf-8') as g:
            data = g.readlines()
        wavdir = [x.split('|')[0].strip() for x in data]
        speaker = [x.split('|')[2].strip() for x in data]

        speaker_dict = self.hp.data.speakers
        n = len(speaker_dict)
        speaker_to_idx = {spk: idx for idx, spk in enumerate(speaker_dict)}

        squares = [0. for i in range(n)]
        means = [0. for i in range(n)]
        frame_count = [0 for i in range(n)]

        for i, fpath in enumerate(tqdm.tqdm(wavdir)):
            spk_idx = speaker_to_idx[speaker[i]]
            square, sum, length = self.process_speaker(os.path.join(self.hp.data.train_dir, fpath),
                                                       self.hp.audio.sampling_rate, self.hp.audio.filter_length,
                                                       self.hp.audio.hop_length, self.hp.audio.f0_min,
                                                       self.hp.audio.f0_max, self.hp.audio.harm_thresh)
            squares[spk_idx] += square
            means[spk_idx] += sum
            frame_count[spk_idx] += length

        result = []
        for i in range(n):
            u = []
            u.append(speaker_dict[i])
            if frame_count[i] == 0:
                avg = 0.0
                avg_sq = 0.0
            else:
                avg = means[i] / frame_count[i]
                avg_sq = squares[i] / frame_count[i]
            u.append(avg)
            u.append(math.sqrt(avg_sq - avg**2))
            result.append(u)

        self.write_metadata(result, self.output_filename)
