import librosa
import soundfile as sf
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from tqdm import tqdm

def extract_audio_from_video(dir_name, save_dir):
    for i in tqdm(sorted(os.listdir(dir_name + '/'))):
        file_name = i.split(".")[0]
        output_path = save_dir + '/'+file_name+'.wav' 
        audio, sr = librosa.load(dir_name + "/" +i, sr=16000)
        sf.write(output_path, audio, sr)

if __name__ == '__main__':
    dataset_dir = '/project/msoleyma_1026/CMU-MOSEI/MOSEI/Videos/Segmented/Combined'
    output_dir = '/project/msoleyma_1026/mosei_edit/audio_seg'
    extract_audio_from_video(dataset_dir, output_dir)