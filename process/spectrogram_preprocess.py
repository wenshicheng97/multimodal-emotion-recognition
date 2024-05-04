import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import librosa
import numpy as np
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

def melspectrogram_process(dir_name, outputnpz_name, outputimg_name):
    for i in tqdm(sorted(os.listdir(dir_name + '/'))):
        if (i=='nohup.out'):
            continue
        file_name = i.split(".")[0]
        output_npz = outputnpz_name + '/'+file_name+'.npy'     
        y, sr = librosa.load(dir_name + "/" +i, sr=16000)
        data = librosa.feature.melspectrogram(y=y, sr=sr)
        data_trans = np.transpose(data)
        np.save(output_npz, data_trans)
        # np.savez(output_npz, audio = data_trans)

        # output_img = outputimg_name + '/'+file_name+'.png'
        # fig = plt.figure(frameon=False)
        # plt.ioff()
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        # fig.add_axes(ax)     
        # fig.show(librosa.display.specshow(librosa.power_to_db(data,ref=np.max), y_axis='mel', fmax=8000, x_axis='time'))
        # fig.savefig(output_img, dpi = 200) 
        # plt.close(fig)


if __name__ == '__main__':
    dataset_dir = '/project/msoleyma_1026/mosei_edit/audio_seg'
    outputnpz_dir = '/project/msoleyma_1026/mosei_edit/audio_seg_npz'
    outputimg_dir = '/scratch1/rliu0866/mosei_edit/audio_img'
    melspectrogram_process(dataset_dir, outputnpz_dir, outputimg_dir)
