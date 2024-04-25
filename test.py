import torch, os
from torchvision.io import read_video
path = '/project/msoleyma_1026/CMU-MOSEI/MOSEI/Videos/Segmented/Combined'
file1 = '_UNQDdiAbWI_1.mp4'
file2 = '_UNQDdiAbWI_0.mp4'

frames, _, _ = read_video(str(os.path.join(path, file1)), output_format="TCHW")
print(f'Frames: {frames.shape}')
frames, _, _ = read_video(str(os.path.join(path, file2)), output_format="TCHW")
print(f'Frames: {frames.shape}')