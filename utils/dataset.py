import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
from pathlib import Path
import numpy as np

def cremad_collate_fn(batch):
    seq_length = [item['seq_length'] for item in batch]
    gaze_angle = pad_sequence([item['gaze_angle'] for item in batch], batch_first=True)
    gaze_landmarks = pad_sequence([item['gaze_landmarks'] for item in batch], batch_first=True)
    face_landmarks = pad_sequence([item['face_landmarks'] for item in batch], batch_first=True)
    spectrogram = pad_sequence([item['spectrogram'] for item in batch], batch_first=True)
    audio = pad_sequence([item['audio'] for item in batch], batch_first=True)
    au17 = pad_sequence([item['au17'] for item in batch], batch_first=True)
    au18 = pad_sequence([item['au18'] for item in batch], batch_first=True)
    label = torch.tensor([item['label'] for item in batch], dtype=torch.long)

    return {
        'seq_length': torch.tensor(seq_length, dtype=torch.long),
        'gaze_angle': gaze_angle,
        'gaze_landmarks': gaze_landmarks,
        'face_landmarks': face_landmarks,
        'spectrogram': spectrogram,
        'audio': audio,
        'au17': au17,
        'au18': au18,
        'label': label }


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
        spectrogram = torch.tensor(data_loaded['spectrogram'], dtype=torch.float32)
        audio = torch.tensor(data_loaded['audio'], dtype=torch.float32)
        au17 = torch.tensor(data_loaded['au17'], dtype=torch.float32)
        au18 = torch.tensor(data_loaded['au18'], dtype=torch.float32)
        
        label = self.emotion_dict[str(data_loaded['label'])]
        seq_length = gaze_angle.shape[0]
        return {
            'seq_length': seq_length,
            'gaze_angle': gaze_angle,
            'gaze_landmarks': gaze_landmarks,
            'face_landmarks': face_landmarks,
            'spectrogram': spectrogram,
            'audio': audio,
            'au17': au17,
            'au18': au18,
            'label': label }



def get_dataloader(data, batch_size):
    if data == 'cremad':
        dataset = CREMADDataset('/scratch1/wenshich/crema-d/batch')

        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size,val_size,test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=cremad_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=cremad_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=cremad_collate_fn)
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloader('cremad', 5)
    for batch in train_loader:
        print('gaze_angle:', batch['gaze_angle'].shape)
        print('gaze_landmarks:', batch['gaze_landmarks'].shape)
        print('face_landmarks:', batch['face_landmarks'].shape)
        print('spectrogram:', batch['spectrogram'].shape)
        print('audio:', batch['audio'].shape)
        print('au17:', batch['au17'].shape)
        print('au18:', batch['au18'].shape)
        print('label:', batch['label'].shape)
        print('seq_length:', batch['seq_length'])
        break
