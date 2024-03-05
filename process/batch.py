from pathlib import Path
import pandas as pd
import numpy as np

cremad_face_path = 'data/CREMA-D/CREMA-D_openface'
cremad_audio_path = 'data/CREMA-D/CREMA-D_spectrogram'
batch_folder = 'batch/batch_v3'

def face_to_batch(face_path, audio_path, target_folder):
    face_path = Path(face_path)
    audio_path = Path(audio_path)
    target_folder = Path(target_folder)
    target_folder.mkdir(exist_ok=True)
    count = 0

    for index, file_path in enumerate(face_path.iterdir()):
        face_df = pd.read_csv(file_path)
        gaze_angle_indices = [' gaze_angle_x', ' gaze_angle_y']
        
        gaze_landmarks_indices = [f' eye_lmk_x_{i}' for i in range(56)] + [f' eye_lmk_y_{i}' for i in range(56)]
        face_landmarks_indices = [f' x_{i}' for i in range(68)] + [f' y_{i}' for i in range(68)]
        gaze_angle = face_df[gaze_angle_indices].values
        gaze_landmarks = face_df[gaze_landmarks_indices].values
        face_landmarks = face_df[face_landmarks_indices].values

        label = file_path.stem.split('_')[2]

        frame_length = gaze_angle.shape[0]
        
        if frame_length <= 10:
            print(f'''{file_path.stem} has less than 10 frames''')
            continue

        audio_array = np.load(audio_path / f'''{file_path.stem}.npz''')['audio']
        diff = gaze_angle.shape[0] - audio_array.shape[0]
        if diff > 0:
            count += 1
            continue
        audio_array = audio_array[:frame_length]
        np.savez(f'{batch_folder}/{file_path.stem}.npz', 
                gaze_angle=gaze_angle, 
                gaze_landmarks=gaze_landmarks, 
                face_landmarks=face_landmarks, 
                audio=audio_array, 
                label=label)

if __name__ == '__main__':
    face_to_batch(cremad_face_path, cremad_audio_path, batch_folder)
