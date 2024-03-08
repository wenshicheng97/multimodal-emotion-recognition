import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import librosa
import numpy as np
import sys
from tqdm import tqdm
import torch
import torchaudio

def melspectrogram_process(dir_name, outputdir_name):
    for i in tqdm(sorted(os.listdir(dir_name + '/'))):
        if (i=='.*'):
            break
        file_name = i.split(".")[0]
        output_path = outputdir_name + '/' +file_name+'.npz'     
        y, sr = librosa.load(str(dir_name + '/' +i), sr=16000)
        data = librosa.feature.melspectrogram(y=y, sr=sr)
        data_trans = np.transpose(data)
        np.savez(output_path, audio = data_trans, label = str(i))


def audio_extraction(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    return waveform.squeeze(0).numpy()


if __name__ == '__main__':
    dataset_dir = 'data/CREMA-D/AudioWAV'
    output_dir = 'res'
    melspectrogram_process(dataset_dir, output_dir)
