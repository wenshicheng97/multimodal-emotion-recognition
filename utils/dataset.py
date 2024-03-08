import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
from pathlib import Path
import numpy as np

def cremad_collate_fn(batch):
    seq_length = [item['seq_length'] for item in batch ]
    gaze_angle = [item['gaze_angle'] for item in batch ]
    gaze_landmarks = [item['gaze_landmarks'] for item in batch ]
    face_landmarks = [item['face_landmarks'] for item in batch ]
    audio = [item['audio'] for item in batch ]
    labels = [item['label'] for item in batch ]

    gaze_angle_padded = pad_sequence(gaze_angle, batch_first=True, padding_value=0)
    gaze_landmarks_padded = pad_sequence(gaze_landmarks, batch_first=True, padding_value=0)
    face_landmarks_padded = pad_sequence(face_landmarks, batch_first=True, padding_value=0)
    audio_padded = pad_sequence(audio, batch_first=True, padding_value=0)

    labels = torch.tensor(labels, dtype=torch.long)
    seq_length = torch.tensor(seq_length, dtype=torch.int)

    return {
        'seq_length': seq_length,
        'gaze_angle': gaze_angle_padded,
        'gaze_landmarks': gaze_landmarks_padded,
        'face_landmarks': face_landmarks_padded,
        'audio': audio_padded,
        'label': labels }


class CREMADDataset(Dataset):
    
    def __init__(self, data_path):
        self.path = Path(data_path)
        self.files = []
        self.emotion_dict = {
            'ANG': 0,
            'DIS': 1,
            'FEA': 2,
            'HAP': 3,
            'NEU': 4,
            'SAD': 5 }
        for file in self.path.glob('*.npz'):
            self.files.append(file)

    
    def __len__(self):
        return len(self.files)

    
    def __getitem__(self, idx):
        file = self.files[idx]
        data_loaded = np.load(file)

        gaze_angle = torch.tensor(data_loaded['gaze_angle'], dtype=torch.float32)
        gaze_landmarks = torch.tensor(data_loaded['gaze_landmarks'], dtype=torch.float32)
        face_landmarks = torch.tensor(data_loaded['face_landmarks'], dtype=torch.float32)
        audio = torch.tensor(data_loaded['audio'], dtype=torch.float32)
        
        label = self.emotion_dict[str(data_loaded['label'])]
        seq_length = gaze_angle.shape[0]
        return {
            'seq_length': seq_length,
            'gaze_angle': gaze_angle,
            'gaze_landmarks': gaze_landmarks,
            'face_landmarks': face_landmarks,
            'audio': audio,
            'label': label }



def get_dataloader(data, batch_size):
    if data == 'cremad':
        dataset = CREMADDataset('batch')

        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size,val_size,test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=cremad_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=cremad_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=cremad_collate_fn)
    return train_loader, val_loader, test_loader
