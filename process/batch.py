from pathlib import Path
import pandas as pd
import numpy as np
from process.audio_preprocess import audio_extraction
import torchaudio

cremad_face_path = '/scratch1/wenshich/CREMA-D_openface'
cremad_spectrogram_path = '/scratch1/wenshich/CREMA-D_spectrogram'
cremad_audio_path = '/project/jonmay_231/wenshich/emotion_recognition/CREMA-D/AudioWAV'
batch_folder = 'batch/'

def write_to_batch(face_path, audio_path, spectrogram_path, target_folder):
    face_path = Path(face_path)
    spectrogram_path = Path(spectrogram_path)
    audio_path = Path(audio_path)
    target_folder = Path(target_folder)
    target_folder.mkdir(exist_ok=True, parents=True)
    count = 0

    for index, file_path in enumerate(face_path.iterdir()):
        print(f'Processing {index+1}: {file_path.stem}')
        face_df = pd.read_csv(file_path)
        gaze_angle_indices = [' gaze_angle_x', ' gaze_angle_y']
        
        gaze_landmarks_indices = [f' eye_lmk_x_{i}' for i in range(56)] + [f' eye_lmk_y_{i}' for i in range(56)]
        face_landmarks_indices = [f' x_{i}' for i in range(68)] + [f' y_{i}' for i in range(68)]
        au17_indices = [f' AU{i:02d}_r' for i in [1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45]]
        au18_indices = [f' AU{i:02d}_c' for i in [1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,28,45]]


        gaze_angle = face_df[gaze_angle_indices].values
        gaze_landmarks = face_df[gaze_landmarks_indices].values
        face_landmarks = face_df[face_landmarks_indices].values
        au17 = face_df[au17_indices].values
        au18 = face_df[au18_indices].values

        label = file_path.stem.split('_')[2]

        frame_length = gaze_angle.shape[0]
        
        if frame_length <= 10 or frame_length > 1000:
            continue

        spectrogram = np.load(spectrogram_path / f'{file_path.stem}.npz')['audio']
        diff = frame_length - spectrogram.shape[0]
        if diff > 0:
            count += 1
            continue
        spectrogram = spectrogram[:frame_length]

        audio = audio_extraction(audio_path / f'{file_path.stem}.wav')
        
        np.savez(f'{batch_folder}/{file_path.stem}.npz', 
                gaze_angle=gaze_angle, 
                gaze_landmarks=gaze_landmarks, 
                face_landmarks=face_landmarks, 
                spectrogram=spectrogram,
                audio=audio,
                au17=au17,
                au18=au18,
                label=label)

if __name__ == '__main__':
    write_to_batch(cremad_face_path, cremad_audio_path, cremad_spectrogram_path, batch_folder)
