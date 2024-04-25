import torch, torchvision, os, yaml
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
import numpy as np
from utils.utils import audio_extraction, padding_video
from transformers import Wav2Vec2Processor

# for fine-tune
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

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index: int):
        # get label
        label_str = os.path.basename(self.video_files[index]).split('_')[2]
        label = self.emotion_dict[label_str]

        # process audio
        audio_path = self.audio_files[index]
        audio = audio_extraction(audio_path, self.processor)

        # process video
        video_path = self.video_files[index]
        reader = torchvision.io.VideoReader(video_path)
        frames_list = []

        for frame in reader:
            frames_list.append(frame['data'])

        video = torch.stack(frames_list) / 255

        if video.shape[0] % self.clip_frames != 0:
            target_frames = video.shape[0] + self.clip_frames - video.shape[0] % self.clip_frames
            video = padding_video(video, target_frames, "same")
        
        video = video.reshape(-1, self.clip_frames, 3, 224, 224).permute(0, 2, 1, 3, 4)
        
        video = video[::self.temporal_sample_rate]
        num_segments = torch.tensor(video.shape[0], dtype=torch.long)
        
        return {'num_seg': num_segments,
                'video': video, 
                'audio': audio, 
                'label': torch.tensor(label, dtype=torch.long)}


# linear probing
class CREMADFeatures(Dataset):

    def __init__(self,
        video_dir: str,
        audio_dir: str
        ):
        super().__init__()
        self.video_dir = video_dir
        self.audio_dir = audio_dir

        self.video_files = sorted([os.path.join(self.video_dir, video) 
                                   for video in os.listdir(self.video_dir) 
                                   if video.endswith('.npz')])
        
        self.audio_files = sorted([os.path.join(self.audio_dir, audio) 
                                   for audio in os.listdir(self.audio_dir) 
                                   if audio.endswith('.wav')])
        
        assert len(self.video_files) == len(self.audio_files), 'Number of video files and audio files do not match!'


        self.emotion_dict = {
            'ANG': 0,
            'DIS': 1,
            'FEA': 2,
            'HAP': 3,
            'NEU': 4,
            'SAD': 5
        }

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index: int):
        # get label
        label_str = os.path.basename(self.video_files[index]).split('_')[2]
        label = self.emotion_dict[label_str]

        # process audio
        audio_path = self.audio_files[index]
        audio = audio_extraction(audio_path, self.processor)

        # process video
        video_path = self.video_files[index]
        video = np.load(video_path)['features']
        video = torch.tensor(video, dtype=torch.float32)
        video_seq_length = video.shape[0]

        return {'video': video, 
                'audio': audio, 
                'seq_length': torch.tensor(video_seq_length, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long)}

    
def cremad_fine_tune(batch):
    num_seg = [item['num_seg'] for item in batch]
    video = [item['video'] for item in batch]
    audio = pad_sequence([item['audio'] for item in batch], batch_first=True)
    label = [item['label'] for item in batch]

    return {'num_seg': torch.tensor(num_seg, dtype=torch.long),
            'video': torch.cat(video, dim=0), 
            'audio': audio, 
            'label': torch.tensor(label, dtype=torch.long)}


def cremad_linear_probing(batch):
    video = pad_sequence([item['video'] for item in batch], batch_first=True)
    audio = pad_sequence([item['audio'] for item in batch], batch_first=True)
    seq_length = [item['seq_length'] for item in batch]
    label = [item['label'] for item in batch]

    return {'video': video,
            'audio': audio, 
            'seq_length': torch.tensor(seq_length, dtype=torch.long), 
            'label': torch.tensor(label, dtype=torch.long)}


def get_dataloader(data, batch_size, fine_tune=False):
    with open('cfgs/path.yaml', 'r') as f:
        path_config = yaml.safe_load(f)

    if data == 'cremad':
        if fine_tune:
            video_path = path_config['dataset']['cremad']['video']['ft']
            audio_path = path_config['dataset']['cremad']['audio']
            dataset = CREMADDataset(video_path, audio_path, clip_frames=16, temporal_sample_rate=2)
            collate_fn = cremad_fine_tune
            
        else:
            video_path = path_config['dataset']['cremad']['video']['lp']
            audio_path = path_config['dataset']['cremad']['audio']
            dataset = CREMADFeatures(video_path, audio_path)
            collate_fn = cremad_linear_probing

        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader



if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloader('cremad', 2, fine_tune=True)
    for batch in train_loader:
        print(batch['video'].shape, batch['label'], batch['audio'].shape, batch['num_seg'])
        a = batch['video'][batch['num_seg']]
        print(a.shape)
        break
