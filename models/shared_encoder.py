import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel, BertModel
from types import SimpleNamespace


class SharedEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.modalities = kwargs['modalities']
        self.n_classes = kwargs['n_classes']
        self.dropout = kwargs['drop_out']
        self.scale_weight = kwargs['scale_weight']
        
        self.hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        for param in self.hubert.parameters():
            param.requires_grad = False
        self.hubert_dim = self.hubert.config.hidden_size
        self.marlin_dim = kwargs['marlin_dim']
        
        self.projector_dim = kwargs['proj_dim']
        self.encoder_dim = kwargs['encoder_dim']

        projectors = [
            nn.Sequential(nn.Linear(self.hubert_dim, self.projector_dim), nn.ReLU(), nn.Dropout(self.dropout)),
            nn.Sequential(nn.Linear(self.marlin_dim, self.projector_dim), nn.ReLU(), nn.Dropout(self.dropout))
        ]

        decoders = [
            nn.Linear(self.encoder_dim, self.hubert_dim), 
            nn.Linear(self.encoder_dim, self.marlin_dim)
        ]

        unimodal_classifiers = [nn.Linear(self.encoder_dim, self.n_classes) for _ in range(2)]

        if kwargs['data'] == 'mosei':
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            for param in self.bert.parameters():
                param.requires_grad = False
            self.bert_dim = self.bert.config.hidden_size
            projectors.append(nn.Sequential(nn.Linear(self.bert_dim, self.projector_dim), nn.ReLU(), nn.Dropout(self.dropout)))
            decoders.append(nn.Linear(self.encoder_dim, self.bert_dim))
            unimodal_classifiers.append(nn.Linear(self.encoder_dim, self.n_classes))

        self.projectors = nn.ModuleList(projectors)
        self.decoders = nn.ModuleList(decoders)
        self.unimodal_classifiers = nn.ModuleList(unimodal_classifiers)

        self.shared_encoder = nn.Sequential(
            nn.Linear(self.projector_dim, self.encoder_dim * 2),
            nn.ReLU(),
            nn.Linear(self.encoder_dim * 2, self.encoder_dim)
        )

        self.multimodal_classifier = nn.Sequential(
            nn.Linear(self.encoder_dim * self.modalities, self.n_classes)
        )

        self.reconstruction_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self, batch, mode='multimodal'):
        audio_input = batch.get('audio', None)
        video_input = batch.get('video', None)
        text_input = batch.get('text', None)
        label = batch['label']
        encoded_list = []
        loss_accumulator = torch.tensor(0.0, device=next(self.shared_encoder.parameters()).device, dtype=next(self.shared_encoder.parameters()).dtype)

        if audio_input is not None and mode in ['audio', 'multimodal']:
            audio_input = self.hubert(audio_input)[0].mean(dim=1)
            audio_loss, audio_logits, audio_encoded = self._unimodal_forward(0, audio_input, label)
            loss_accumulator += audio_loss
            encoded_list.append(audio_encoded)

            if mode == 'audio':
                return SimpleNamespace(label=label,
                                    loss=audio_loss,
                                    logits=audio_logits)
        
        if video_input is not None and mode in ['video', 'multimodal']:
            video_input = video_input.mean(dim=1)
            video_loss, video_logits, video_encoded = self._unimodal_forward(1, video_input, label)
            loss_accumulator += video_loss
            encoded_list.append(video_encoded)

            if mode == 'video':
                return SimpleNamespace(label=label,
                                    loss=video_loss,
                                    logits=video_logits)
        
        if text_input is not None and hasattr(self, 'bert') and mode in ['text', 'multimodal']:
            attention_mask = (text_input > 0).int()
            text_input = self.bert(text_input, attention_mask=attention_mask).last_hidden_state[:,0,:]
            text_loss, text_logits, text_encoded = self._unimodal_forward(2, text_input, label)
            loss_accumulator += text_loss
            encoded_list.append(text_encoded)

            if mode == 'text':
                return SimpleNamespace(label=label,
                                    loss=text_loss,
                                    logits=text_logits)

        if mode == 'multimodal':
            multimodal_input = torch.cat(encoded_list, dim=1)
            multimodal_logits = self.multimodal_classifier(multimodal_input)
            multimodal_loss = self.classification_loss(multimodal_logits, label)
            loss_accumulator += multimodal_loss
            return SimpleNamespace(label=label, loss=loss_accumulator, logits=multimodal_logits)

        
    def _unimodal_forward(self, modality_idx, x, y):
        proj = self.projectors[modality_idx](x)
        encoded = self.shared_encoder(proj)
        decoded = self.decoders[modality_idx](encoded)
        reconstruction_loss = self.reconstruction_loss(decoded, x)
        logits = self.unimodal_classifiers[modality_idx](encoded)
        classification_loss = self.classification_loss(logits, y)
        return self.scale_weight * reconstruction_loss + classification_loss, logits, encoded
    
    def train_mode(self, mode='multimodal'):
        if mode == 'multimodal':
            return FreezeModel([self.projectors, self.shared_encoder, self.decoders])


class FreezeModel:
    def __init__(self, modules):
        if not isinstance(modules, list):
            modules = [modules]
        self.modules = modules
        self.original_states = []

    def __enter__(self):
        for module in self.modules:
            module_state = {}
            for param in module.parameters():
                module_state[param] = param.requires_grad
                param.requires_grad = False
            self.original_states.append(module_state)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for module, state_dict in zip(self.modules, self.original_states):
            for param, original in state_dict.items():
                param.requires_grad = original
        return False