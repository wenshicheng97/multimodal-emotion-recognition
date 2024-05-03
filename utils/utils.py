import torch, torchaudio, torchvision
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
from torch.nn import functional as F
from torchaudio.sox_effects import apply_effects_file
import pandas as pd


def audio_extraction(audio_path, processor):
    # waveform, sample_rate = torchaudio.load(audio_path)
    # if sample_rate != 16000:
    #     resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    #     waveform = resampler(waveform)
    # return waveform.squeeze(0)
    waveform, sample_rate = apply_effects_file(audio_path, effects=[["rate", "16000"]])
    inputs = processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt", padding="longest", truncation=True, max_length=96000)
    return inputs.input_values.squeeze(0)

def text_tokenize(text_path, file_name, processor):
    text_file = pd.read_csv(text_path)
    text_seg = text_file[text_file["name"] == file_name]["text"].to_list()[0]
    inputs = processor.encode(text_seg, padding = "longest", return_tensors="pt")
    return inputs.squeeze(0)


def read_video(path: str, channel_first: bool = True):
    video, _, _ = torchvision.io.read_video(path)
    if channel_first:
        video = rearrange(video, 'T H W C -> T C H W')
    return video


def sample_indexes(total_frames: int, n_frames: int, temporal_sample_rate: int) -> torch.Tensor:
    try:
        start_ind = torch.randint(0, total_frames - (n_frames * temporal_sample_rate) + 1, ())
    except RuntimeError as e:
        print(f"total_frames: {total_frames}, n_frames: {n_frames}, temporal_sample_rate: {temporal_sample_rate}")
        raise e
    return torch.arange(n_frames) * temporal_sample_rate + start_ind


def padding_video(tensor: torch.Tensor, target: int, padding_method: str = "zero", padding_position: str = "tail") -> torch.Tensor:
    t, c, h, w = tensor.shape
    padding_size = target - t

    pad = _get_padding_pair(padding_size, padding_position)

    if padding_method == "zero":
        return F.pad(tensor, pad=[0, 0, 0, 0, 0, 0] + pad)
    elif padding_method == "same":
        tensor = rearrange(tensor, "t c h w -> c h w t")
        tensor = F.pad(tensor, pad=pad + [0, 0], mode="replicate")
        return rearrange(tensor, "c h w t -> t c h w")
    else:
        raise ValueError("Wrong padding method. It should be zero or tail or average.")
    

def _get_padding_pair(padding_size: int, padding_position: str) -> list[int]:
    if padding_position == "tail":
        pad = [0, padding_size]
    elif padding_position == "head":
        pad = [padding_size, 0]
    elif padding_position == "average":
        padding_head = padding_size // 2
        padding_tail = padding_size - padding_head
        pad = [padding_head, padding_tail]
    else:
        raise ValueError("Wrong padding position. It should be zero or tail or average.")
    return pad


def window_transform(batch, lengths, window_size, stride):
    batch_size = batch.size(0)
    transformed = []
    seq_length = []

    for i in range(batch_size):
        current_length = lengths[i]
        current_sequence = batch[i, :current_length, :]
        transformed_sequence = []

        for start in range(0, current_length, stride):
            end = start + window_size
            if end > current_length:
                end = current_length
                start = end - window_size

            segment = current_sequence[start:end, :].view(-1)
            transformed_sequence.append(segment)

        transformed.append(torch.stack(transformed_sequence))
        seq_length.append(len(transformed_sequence))

    # padded_transformed: (batch_size, max_seq_len, feature_size * window)
    padded_transformed = pad_sequence(transformed, batch_first=True, padding_value=0)
    seq_length = torch.tensor(seq_length, dtype=torch.int)

    return padded_transformed, seq_length

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


class CREMAD_Normalizer:
    def __init__(self = None, loader = None):
        self.loader = loader
        self.normalizer = {
            'gaze_angle': dict(),
            'gaze_landmarks': dict(),
            'face_landmarks': dict(),
            'audio': dict() }
        self._normalize()

    
    def _normalize(self):
        total_task = len(self.normalizer)
        min_x = torch.full((total_task,), float('inf'))
        max_x = torch.full((total_task,), float('-inf'))
        min_y = torch.full((total_task,), float('inf'))
        max_y = torch.full((total_task,), float('-inf'))
        for batch in self.loader:
            for task_idx, task in enumerate(self.normalizer.keys()):
                current = batch[task]
                feature_size = current.size(-1)

                x_coord = current[..., :feature_size // 2]
                y_coord = current[..., feature_size // 2:]

                current_min_x = torch.min(x_coord)
                current_max_x = torch.max(x_coord)
                current_min_y = torch.min(y_coord)
                current_max_y = torch.max(y_coord)

                min_x[task_idx] = torch.min(min_x[task_idx], current_min_x)
                max_x[task_idx] = torch.max(max_x[task_idx], current_max_x)
                min_y[task_idx] = torch.min(min_y[task_idx], current_min_y)
                max_y[task_idx] = torch.max(max_y[task_idx], current_max_y)

        for task_idx, task in enumerate(self.normalizer.keys()):
            current_normalizer = {
                'x': {
                    'min': min_x[task_idx],
                    'max': max_x[task_idx] },
                'y': {
                    'min': min_y[task_idx],
                    'max': max_y[task_idx] } }
            self.normalizer[task] = current_normalizer

    
    def minmax_normalize(self, tensor, task):
        normalized = tensor.clone()

        feature_size = normalized.size(-1)
        x_coords = normalized[..., :feature_size // 2]
        y_coords = normalized[..., feature_size // 2:]

        normalized_x_coords = (x_coords - self.normalizer[task]['x']['min']) / (self.normalizer[task]['x']['max'] - self.normalizer[task]['x']['min'])
        normalized_y_coords = (y_coords - self.normalizer[task]['y']['min']) / (self.normalizer[task]['y']['max'] - self.normalizer[task]['y']['min'])

        normalized[..., :feature_size // 2] = normalized_x_coords
        normalized[..., feature_size // 2:] = normalized_y_coords
        return normalized

    
    def minmax_denormalize(self, tensor, task):
        demoralized = tensor.clone()
        feature_size = demoralized.size(-1)

        x_coords = demoralized[..., :feature_size // 2]
        y_coords = demoralized[..., feature_size // 2:]

        denormalized_x_coords = x_coords * (self.normalizer[task]['x']['max'] - self.normalizer[task]['x']['min']) + self.normalizer[task]['x']['min']
        denormalized_y_coords = y_coords * (self.normalizer[task]['y']['max'] - self.normalizer[task]['y']['min']) + self.normalizer[task]['y']['min']

        demoralized[..., :feature_size // 2] = denormalized_x_coords
        demoralized[..., feature_size // 2:] = denormalized_y_coords
        return demoralized
