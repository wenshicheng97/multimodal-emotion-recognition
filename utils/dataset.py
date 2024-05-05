# import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
import torch, torchvision, os, yaml
from torch.utils.data import Dataset, DataLoader, Subset, random_split, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from utils.utils import audio_extraction, padding_video, text_tokenize
from transformers import Wav2Vec2Processor, BertTokenizerFast
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class MultimodalDataset(Dataset):
    def __init__(self,
                 video_dir: str,
                 audio_dir: str,
                 audio_pretrained = "facebook/hubert-large-ls960-ft"):
        super().__init__()
        self.video_dir = video_dir
        self.audio_dir = audio_dir

        self.audio_processor = Wav2Vec2Processor.from_pretrained(audio_pretrained)
    

class CREMAD(MultimodalDataset):
    def __init__(self, video_dir: str, audio_dir: str, audio_pretrained="facebook/hubert-large-ls960-ft"):
        super().__init__(video_dir, audio_dir, audio_pretrained)
        self.emotion_dict = {
            'ANG': 0,
            'DIS': 1,
            'FEA': 2,
            'HAP': 3,
            'NEU': 4,
            'SAD': 5
        }
        self.video_files = sorted([os.path.join(self.video_dir, video) 
                                   for video in os.listdir(self.video_dir)])
        
        self.audio_files = sorted([os.path.join(self.audio_dir, audio) 
                                   for audio in os.listdir(self.audio_dir)])
        
        assert len(self.video_files) == len(self.audio_files), 'Number of video files and audio files do not match!'

    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, index: int):
        # get label
        label_str = os.path.basename(self.video_files[index]).split('_')[2]
        label = self.emotion_dict[label_str]
        label = torch.tensor(label, dtype=torch.long)

        # process audio
        audio_path = self.audio_files[index]
        audio = audio_extraction(audio_path, self.audio_processor)

        return audio, label
    

# for fine-tune
class CREMADDataset(CREMAD):
    def __init__(self, 
                 video_dir: str, 
                 audio_dir: str,
                 clip_frames: int,
                 temporal_sample_rate: int,
                 audio_pretrained="facebook/hubert-large-ls960-ft"):
        super().__init__(video_dir, audio_dir, audio_pretrained)
        self.clip_frames = clip_frames
        self.temporal_sample_rate = temporal_sample_rate


    def __getitem__(self, index: int):
        audio, label = super().__getitem__(index)

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
                'label': label}


# linear probing
class CREMADFeatures(CREMAD):
    def __init__(self, video_dir: str, 
                 audio_dir: str, 
                 audio_pretrained="facebook/hubert-large-ls960-ft"):
        super().__init__(video_dir, audio_dir, audio_pretrained)

    def __getitem__(self, index: int):
        audio, label = super().__getitem__(index)

        # process video
        video_path = self.video_files[index]
        video = np.load(video_path)['features']
        video = torch.tensor(video, dtype=torch.float32)
        video_seq_length = video.shape[0]

        return {'video': video, 
                'audio': audio, 
                'seq_length': torch.tensor(video_seq_length, dtype=torch.long),
                'label': label}
    

class MOSEI(MultimodalDataset):
    def __init__(self, video_dir: str, 
                 audio_dir: str,
                 text_file: str,
                 annotation_file: str,
                 audio_pretrained="facebook/hubert-large-ls960-ft",
                 text_pretrained="bert-base-cased"):
        super().__init__(video_dir, audio_dir, audio_pretrained)
        annotations = pd.read_csv(annotation_file, dtype={'name': str})
        self.file_names = annotations['name'].values.tolist()
        self.labels = annotations['label'].values.tolist()
        self.text_processor = BertTokenizerFast.from_pretrained(text_pretrained)
        self.text_file = text_file

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)

        audio_path = os.path.join(self.audio_dir, self.file_names[index]+'.wav')
        audio = audio_extraction(audio_path, self.audio_processor)

        text = text_tokenize(self.text_file, self.file_names[index], self.text_processor)
        return audio, text, label
    

class MOSEIDataset(MOSEI):
    def __init__(self, video_dir: str, 
                 audio_dir: str,
                 text_file: str,
                 annotation_file: str,
                 clip_frames: int,
                 audio_pretrained="facebook/hubert-large-ls960-ft",
                 text_pretrained="bert-base-cased"):
        super().__init__(video_dir, audio_dir, text_file, annotation_file, audio_pretrained, text_pretrained)
        self.clip_frames = clip_frames

    def __getitem__(self, index: int):
        audio, text, label = super().__getitem__(index)

        video_path = os.path.join(self.video_dir, self.file_names[index]+'.mp4')
        reader = torchvision.io.VideoReader(video_path)
        frames_list = []

        for frame in reader:
            frames_list.append(frame['data'])

        video = torch.stack(frames_list) / 255

        if video.shape[0] < self.clip_frames:
            video = padding_video(video, self.clip_frames, "same")
        else:
            # take evenly spaced frames
            start_idx = 0 if video.shape[0] % self.clip_frames == 0 \
                        else torch.randint(0, video.shape[0] % self.clip_frames, ())
            interval = video.shape[0] // self.clip_frames

            # take 32 frames evenly spaced
            video_indices = torch.arange(start_idx, video.shape[0], interval)[:self.clip_frames]
            video = video[video_indices]

        video = video.reshape(-1, 16, 3, 224, 224).permute(0, 2, 1, 3, 4)
        
        num_segments = torch.tensor(video.shape[0], dtype=torch.long)
        
        return {'num_seg': num_segments,
                'video': video, 
                'audio': audio, 
                'text': text,
                'label': label}
    

class MOSEIFeatures(MOSEI):
    def __init__(self, video_dir: str, 
                 audio_dir: str,
                 text_file: str,
                 annotation_file: str,
                 audio_pretrained="facebook/hubert-large-ls960-ft",
                 text_pretrained="bert-base-cased"):
        super().__init__(video_dir, audio_dir, text_file, annotation_file, audio_pretrained, text_pretrained)

    def __getitem__(self, index: int):
        audio, text, label = super().__getitem__(index)

        # process video
        video_path = os.path.join(self.video_dir, self.file_names[index]+'.npz')
        video = np.load(video_path)['features']
        video = torch.tensor(video, dtype=torch.float32)
        video_seq_length = video.shape[0]

        return {'video': video, 
                'audio': audio, 
                'text': text,
                'seq_length': torch.tensor(video_seq_length, dtype=torch.long),
                'label': label}
    

class MOSEILSTM(Dataset):
    def __init__(self, openface_dir: str,
                 spectrogram_dir: str,
                 annotation_file: str):
        annotations = pd.read_csv(annotation_file, dtype={'name': str})
        self.file_names = annotations['name'].values.tolist()
        self.labels = annotations['label'].values.tolist()
        self.openface_dir = openface_dir
        self.spectrogram_dir = spectrogram_dir

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index: int):
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)

        openface_path = os.path.join(self.openface_dir, self.file_names[index]+'.csv')
        openface = pd.read_csv(openface_path)
        au17_indices = [f' AU{i:02d}_r' for i in [1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45]]
        au18_indices = [f' AU{i:02d}_c' for i in [1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,28,45]]

        au17 = torch.tensor(openface[au17_indices].values, dtype=torch.float32)
        au18 = torch.tensor(openface[au18_indices].values, dtype=torch.float32)

        spectrogram_path = os.path.join(self.spectrogram_dir, self.file_names[index]+'.npy')
        spectrogram = np.load(spectrogram_path)
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)

        return {'au17': au17,
                'au18': au18,
                'seq_length': au17.shape[0], 
                'spectrogram': spectrogram,
                'label': label}


    
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

def mosei_fine_tune(batch):
    num_seg = [item['num_seg'] for item in batch]
    video = [item['video'] for item in batch]
    audio = pad_sequence([item['audio'] for item in batch], batch_first=True)
    text = pad_sequence([item['text'] for item in batch], batch_first=True)
    label = [item['label'] for item in batch]

    return {'num_seg': torch.tensor(num_seg, dtype=torch.long),
            'video': torch.cat(video, dim=0), 
            'audio': audio, 
            'text': text,
            'label': torch.tensor(label, dtype=torch.long)}


def mosei_linear_probing(batch):
    video = pad_sequence([item['video'] for item in batch], batch_first=True)
    audio = pad_sequence([item['audio'] for item in batch], batch_first=True)
    text = pad_sequence([item['text'] for item in batch], batch_first=True)
    seq_length = [item['seq_length'] for item in batch]
    label = [item['label'] for item in batch]

    return {'video': video,
            'audio': audio, 
            'text': text,
            'seq_length': torch.tensor(seq_length, dtype=torch.long), 
            'label': torch.tensor(label, dtype=torch.long)}


def mosei_lstm(batch):
    au17 = pad_sequence([item['au17'] for item in batch], batch_first=True)
    au18 = pad_sequence([item['au18'] for item in batch], batch_first=True)
    spectrogram = pad_sequence([item['spectrogram'] for item in batch], batch_first=True)
    label = [item['label'] for item in batch]
    seq_length = [item['seq_length'] for item in batch]

    return {'au17': au17,
            'au18': au18,
            'spectrogram': spectrogram,
            'label': torch.tensor(label, dtype=torch.long),
            'seq_length': torch.tensor(seq_length, dtype=torch.long)}


def get_dataloader(data, batch_size, fine_tune=False, lstm=False):
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

    elif data == 'mosei':
        if lstm:
            openface_dir = path_config['dataset']['mosei']['openface']
            spectrogram_dir = path_config['dataset']['mosei']['spectrogram']
            annotation_file = path_config['dataset']['mosei']['annotation_openface']
            dataset = MOSEILSTM(openface_dir, spectrogram_dir, annotation_file)
            collate_fn = mosei_lstm
        else:
            annotation_file = path_config['dataset']['mosei']['annotation']
            if fine_tune:
                video_path = path_config['dataset']['mosei']['video']['ft']
                audio_path = path_config['dataset']['mosei']['audio']
                text_path = path_config['dataset']['mosei']['text']
                dataset = MOSEIDataset(video_path, audio_path, text_path, annotation_file=annotation_file, clip_frames=32)
                collate_fn = mosei_fine_tune
                
            else:
                video_path = path_config['dataset']['mosei']['video']['lp']
                audio_path = path_config['dataset']['mosei']['audio']
                text_path = path_config['dataset']['mosei']['text']
                dataset = MOSEIFeatures(video_path, audio_path, text_path, annotation_file=annotation_file)
                collate_fn = mosei_linear_probing

        # split videos with different people into train and val
        # and make sure label balance
        df = pd.read_csv(annotation_file)
        df['video_base_name'] = df['name'].apply(lambda x: '_'.join(x.split('_')[:-1]))
        representative_labels = df.groupby('video_base_name')['label'].first().reset_index()

        df = df.merge(representative_labels, on='video_base_name', suffixes=('', '_representative'))
        train_videos, val_videos = train_test_split(representative_labels, test_size=0.2, stratify=representative_labels['label'])

        train_df = df[df['video_base_name'].isin(train_videos['video_base_name'])]
        val_df = df[df['video_base_name'].isin(val_videos['video_base_name'])]

        train_indices = train_df.index.tolist()
        val_indices = val_df.index.tolist()

        train_dataset, val_dataset = Subset(dataset, train_indices), Subset(dataset, val_indices)

        # handle data imbalance
        train_labels = [dataset.labels[i] for i in train_indices]
        class_sample_counts = torch.bincount(torch.tensor(train_labels))
        weights = 1 / class_sample_counts[train_labels]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
        test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


# if __name__ == '__main__':
    # train_loader, val_loader, test_loader = get_dataloader('mosei', 2, fine_tune=True, lstm=False)

    # for batch in tqdm(train_loader):
    #     print(batch['text'].shape)
    #     print(f'text batch: {batch["text"]}')
    #     break
    # df = pd.read_csv('/project/msoleyma_1026/mosei_edit/mosei_data.csv')
    # openface_dir = '/project/msoleyma_1026/mosei_edit/MOSEI_openface'
    # print(len(df))
    # df['file_exists'] = df['name'].apply(lambda x: os.path.exists(os.path.join(openface_dir, f'{x}.csv')))
    # filtered_df = df[df['file_exists']]

    # # write to csv
    # filtered_df.to_csv('/project/msoleyma_1026/mosei_edit/mosei_data_openface.csv', index=False)
    # filtered_df = pd.read_csv('/project/msoleyma_1026/mosei_edit/mosei_data_openface.csv')
    # print(len(filtered_df))

    # df = pd.read_csv('/project/msoleyma_1026/mosei_edit/mosei_label_common.csv')
    # print(len(df))
    # df['file_exists'] = df['label'].apply(lambda x: x!=0)
    # filtered_df = df[df['file_exists']]
    # filtered_df['label'] = filtered_df['label'].apply(lambda x: x-1)
    # # # write to csv
    # filtered_df.to_csv('/project/msoleyma_1026/mosei_edit/mosei_data.csv', index=False)
    # filtered_df = pd.read_csv('/project/msoleyma_1026/mosei_edit/mosei_data.csv')
    # print(len(filtered_df))
    # print(filtered_df['label'].value_counts())

    # video = torch.stack(frames_list) / 255
    # print(video.shape)
    # target_frames = 32 
    # video = padding_video(video, target_frames, "same")
    # print(video.shape)
    # pass
