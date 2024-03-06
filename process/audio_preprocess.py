import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import librosa
import numpy as np
import sys
from tqdm import tqdm

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

if __name__ == '__main__':
    dataset_dir = 'data/CREMA-D/AudioWAV'
    output_dir = 'res'
    melspectrogram_process(dataset_dir, output_dir)
