import torch
from torch.nn.utils.rnn import pad_sequence

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
