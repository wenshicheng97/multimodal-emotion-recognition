import torch, torchvision
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
from pathlib import Path
import numpy as np
import ffmpeg
from itertools import islice
from utils.utils import audio_extraction, read_video, padding_video, sample_indexes

# def cremad_collate_fn(batch):
#     seq_length = [item['seq_length'] for item in batch]
#     gaze_angle = pad_sequence([item['gaze_angle'] for item in batch], batch_first=True)
#     gaze_landmarks = pad_sequence([item['gaze_landmarks'] for item in batch], batch_first=True)
#     face_landmarks = pad_sequence([item['face_landmarks'] for item in batch], batch_first=True)
#     spectrogram = pad_sequence([item['spectrogram'] for item in batch], batch_first=True)
#     audio = pad_sequence([item['audio'] for item in batch], batch_first=True)
#     au17 = pad_sequence([item['au17'] for item in batch], batch_first=True)
#     au18 = pad_sequence([item['au18'] for item in batch], batch_first=True)
#     label = torch.tensor([item['label'] for item in batch], dtype=torch.long)

#     return {
#         'seq_length': torch.tensor(seq_length, dtype=torch.long),
#         'gaze_angle': gaze_angle,
#         'gaze_landmarks': gaze_landmarks,
#         'face_landmarks': face_landmarks,
#         'spectrogram': spectrogram,
#         'audio': audio,
#         'au17': au17,
#         'au18': au18,
#         'label': label }


# class CREMADDataset(Dataset):
    
#     def __init__(self, data_path):
#         self.path = Path(data_path)
#         self.files = []
#         self.emotion_dict = {
#             'ANG': 0,
#             'DIS': 1,
#             'FEA': 2,
#             'HAP': 3,
#             'NEU': 4,
#             'SAD': 5 }
#         for file in self.path.glob('*.npz'):
#             self.files.append(file)

    
#     def __len__(self):
#         return len(self.files)

    
#     def __getitem__(self, idx):
#         file = self.files[idx]
#         data_loaded = np.load(file)

#         gaze_angle = torch.tensor(data_loaded['gaze_angle'], dtype=torch.float32)
#         gaze_landmarks = torch.tensor(data_loaded['gaze_landmarks'], dtype=torch.float32)
#         face_landmarks = torch.tensor(data_loaded['face_landmarks'], dtype=torch.float32)
#         spectrogram = torch.tensor(data_loaded['spectrogram'], dtype=torch.float32)
#         audio = torch.tensor(data_loaded['audio'], dtype=torch.float32)
#         au17 = torch.tensor(data_loaded['au17'], dtype=torch.float32)
#         au18 = torch.tensor(data_loaded['au18'], dtype=torch.float32)
        
#         label = self.emotion_dict[str(data_loaded['label'])]
#         seq_length = gaze_angle.shape[0]
#         return {
#             'seq_length': seq_length,
#             'gaze_angle': gaze_angle,
#             'gaze_landmarks': gaze_landmarks,
#             'face_landmarks': face_landmarks,
#             'spectrogram': spectrogram,
#             'audio': audio,
#             'au17': au17,
#             'au18': au18,
#             'label': label }

# def get_dataloader(data, batch_size):
#     if data == 'cremad':
#         dataset = CREMADDataset('batch/batch_v4')

#         train_size = int(0.8 * len(dataset))
#         val_size = int(0.1 * len(dataset))
#         test_size = len(dataset) - train_size - val_size
#         train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size,val_size,test_size])

#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=cremad_collate_fn)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=cremad_collate_fn)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=cremad_collate_fn)
#     return train_loader, val_loader, test_loader

import os
from torch.utils.data import Dataset
from marlin_pytorch import Marlin


class CREMADDataset(Dataset):

    def __init__(self,
        video_dir: str,
        audio_dir: str,
        clip_frames: int,
        temporal_sample_rate: int,
        ):
        super().__init__()
        self.video_dir = video_dir
        self.audio_dir = audio_dir

        self.video_files = sorted([os.path.join(self.video_dir, video) 
                                   for video in os.listdir(self.video_dir) 
                                   if video.endswith('.mp4')])
        
        self.audio_files = sorted([os.path.join(self.audio_dir, audio) 
                                   for audio in os.listdir(self.audio_dir) 
                                   if audio.endswith('.wav')])
        
        assert len(self.video_files) == len(self.audio_files), 'Number of video files and audio files do not match!'

        self.clip_frames = clip_frames
        self.temporal_sample_rate = temporal_sample_rate

        self.emotion_dict = {
            'ANG': 0,
            'DIS': 1,
            'FEA': 2,
            'HAP': 3,
            'NEU': 4,
            'SAD': 5
        }

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index: int):
        # get label
        label_str = os.path.basename(self.video_files[index]).split('_')[2]
        label = self.emotion_dict[label_str]

        # process audio
        audio_path = self.audio_files[index]
        audio = audio_extraction(audio_path)

        # process video
        video_path = self.video_files[index]
        probe = ffmpeg.probe(video_path)["streams"][0]
        n_frames = int(probe["nb_frames"])

        if n_frames <= self.clip_frames:
            video = read_video(video_path, channel_first=True).video / 255
            # pad frames to 16
            video = padding_video(video, self.clip_frames, "same")  # (T, C, H, W)
            video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
            
            return video, audio, torch.tensor(label, dtype=torch.long)
        
        elif n_frames <= self.clip_frames * self.temporal_sample_rate:
            # reset a lower temporal sample rate
            sample_rate = n_frames // self.clip_frames
        else:
            sample_rate = self.temporal_sample_rate
        # sample frames
        video_indexes = sample_indexes(n_frames, self.clip_frames, sample_rate)
        reader = torchvision.io.VideoReader(video_path)
        fps = reader.get_metadata()["video"]["fps"][0]
        reader.seek(video_indexes[0].item() / fps, True)
        frames = []
        for frame in islice(reader, 0, self.clip_frames * sample_rate, sample_rate):
            frames.append(frame["data"])
        video = torch.stack(frames) / 255  # (T, C, H, W)
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
        assert video.shape[1] == self.clip_frames, video_path

        return {'video': video, 'audio': audio, 'label': torch.tensor(label, dtype=torch.long)}
    
def cremad_collate_fn(batch):
    # video, audio, label = zip(*batch)
    # padded_audio = pad_sequence(audio, batch_first=True)

    # return torch.stack(video), padded_audio, torch.tensor(label, dtype=torch.long)
    video = [item['video'] for item in batch]
    audio = pad_sequence([item['audio'] for item in batch], batch_first=True)
    label = [item['label'] for item in batch]

    return {'video': torch.stack(video), 
            'audio': audio, 
            'label': torch.tensor(label, dtype=torch.long)}


def get_dataloader(data, batch_size):
    if data == 'cremad':
        video_path = '/home/tangyimi/emotion/data/cremad/cropped_face'
        audio_path = '/home/tangyimi/emotion/data/cremad/AudioWAV'
        dataset = CREMADDataset(video_path, audio_path, clip_frames=16, temporal_sample_rate=2)

        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=cremad_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=cremad_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=cremad_collate_fn)

    return train_loader, val_loader, test_loader



if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloader('cremad', 2)
    for batch in train_loader:
        print(batch['video'].shape, batch['audio'].shape, batch['label'])
        break
