import pandas as pd
import numpy as np
from marlin_pytorch import Marlin
import torch, os

cremad_cropped_face = '/home/tangyimi/emotion/data/cremad/cropped_face'
cremad_marlin_feat_path = '/home/tangyimi/emotion/data/cremad/marlin_face_base'

mosei_labels = '/project/msoleyma_1026/mosei_edit/mosei_label_common.csv'
mosei_seg_video_path = '/project/msoleyma_1026/mosei_edit/video_seg'
mosei_marlin_target = '/project/msoleyma_1026/mosei_edit/marlin_feat'

def marlin_cremad(video_path, target_path, crop_face):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Marlin.clean_cache()
    model = Marlin.from_online("marlin_vit_base_ytf").to(device) # marlin_vit_small_ytf # marlin_vit_large_ytf
    
    for index, filename in enumerate(os.listdir(video_path)):
        print(f'Processing {index+1}: {filename}')
        features = model.extract_video(f'{video_path}/{filename}', crop_face=crop_face)
        features = features.cpu().numpy()
        print(f'features shape: {features.shape}')
        return
        #label = filename.split('_')[2]
        np.savez(f"{target_path}/{filename.split('.')[0]}.npz", features=features,) #label=label)

def marlin_mosei(video_path, label_path, target_path, crop_face):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Marlin.clean_cache()
    model = Marlin.from_online("marlin_vit_base_ytf").to(device) # marlin_vit_small_ytf # marlin_vit_large_ytf
    df = pd.read_csv(label_path)

    for index, row in df.iterrows():
        filename = row['name']
        label = int(row['label'])
        print(f'Processing {index+1}: {filename}')
        features = model.extract_video(f'{video_path}/{filename}.mp4', crop_face=crop_face)
        features = features.cpu().numpy()
        np.savez(f"{target_path}/{filename}.npz", features=features, label=label)
    

if __name__ == '__main__':
    marlin_mosei(mosei_seg_video_path, mosei_labels, mosei_marlin_target, crop_face=False)