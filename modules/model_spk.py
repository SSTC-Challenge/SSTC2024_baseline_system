import torch.nn as nn
import torch

from modules.pooling import AttentiveStatisticsPooling
from modules.front_conformer_mfa import ConformerEncoder as ConformerEncoderMFA


class ConformerMFA(nn.Module):
    def __init__(self, cfg, embedding_size, dropout=0.5, ctc=False, add_feat=0, num_layer=None):
        super().__init__()
        self.cfg = cfg
        self.front = ConformerEncoderMFA(**cfg)
        num_layer = num_layer + add_feat if num_layer is not None else cfg['n_layers'] + add_feat
        front_out = num_layer * cfg['d_model']
        self.norm_utt1 = nn.LayerNorm(front_out)
        self.pool = AttentiveStatisticsPooling(front_out)
        self.norm_utt2 = nn.BatchNorm1d(front_out*2)
        self.bottleneck = nn.Linear(front_out*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        self.ctc = ctc

    def forward(self, x, xlen, add_feat=None, memory=None, layer_distillation=False):
        x = x[:, :, :xlen.max()]

        if memory is not None:
            memory = memory.transpose(1, 2)

        if layer_distillation:
            x, _, x_utt, att_maps = self.front(audio_signal=x, length=xlen, memory=memory, get_att_map=layer_distillation) # x_utt: BxTxD; x: BxDxT
            layer_outputs = x_utt
        else:
            x, _, x_utt = self.front(audio_signal=x, length=xlen, memory=memory, get_att_map=layer_distillation) # x_utt: BxTxD; x: BxDxT

        if add_feat is not None:
            assert self.cfg['mask_prob'] == 0
            x_utt = torch.cat([add_feat.transpose(1, 2), x_utt], dim=-1)
            
        if self.cfg['mask_prob'] > 0 and self.training:
            x_utt[0] = self.norm_utt1(x_utt[0]).transpose(1, 2)
            x_utt[0] = self.pool(x_utt[0])[:, :, 0] # BxTx1 -> BxT
            x_utt[0] = self.norm_utt2(x_utt[0])
            x_utt[0] = self.bottleneck(x_utt[0])

            x_utt[1] = self.norm_utt1(x_utt[1]).transpose(1, 2)
            x_utt[1] = self.pool(x_utt[1])[:, :, 0] # BxTx1 -> BxT
            x_utt[1] = self.norm_utt2(x_utt[1])
            x_utt[1] = self.bottleneck(x_utt[1])

            if self.drop:
                x_utt[0] = self.drop(x_utt[0])
                x_utt[1] = self.drop(x_utt[1])
        else:
            x_utt = torch.cat(x_utt, dim=-1)
            x_utt = self.norm_utt1(x_utt).transpose(1, 2) # BxDxT
            x_utt = self.pool(x_utt)[:, :, 0] # BxTx1 -> BxT
            x_utt = self.norm_utt2(x_utt)
            x_utt = self.bottleneck(x_utt)
            if self.drop:
                x_utt = self.drop(x_utt)

        if self.ctc:
            return x_utt, x
        elif layer_distillation:
            return x_utt, layer_outputs, att_maps
        else:
            return x_utt

class Adaptor(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, embedding_size, dropout=0):
        super().__init__()
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(num_layers)
        ])
        front_out = num_layers * hidden_dim
        self.mfa_norm = nn.LayerNorm(front_out)
        self.pool = AttentiveStatisticsPooling(front_out)
        self.utt_norm = nn.BatchNorm1d(front_out*2)
        self.bottleneck = nn.Linear(front_out*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
    
    def forward(self, x):
        x = torch.cat([self.proj_layers[i](x[i]) for i in range(len(x))], dim=-1)
        x = self.mfa_norm(x).transpose(1, 2)
        x = self.pool(x)[:, :, 0]
        x = self.utt_norm(x)
        x = self.bottleneck(x)
        return self.drop(x) if self.drop else x

class ConformerMFA_MultiTask(nn.Module):
    def __init__(self, cfg, embedding_size, dropout=0.5, real_feat=False):
        super().__init__()
        self.cfg = cfg
        self.front = ConformerEncoderMFA(**cfg)
        self.src_feat = Adaptor(cfg['n_layers'], cfg['d_model'], 128, embedding_size, dropout)
        self.tgt_feat = Adaptor(cfg['n_layers'], cfg['d_model'], 128, embedding_size, dropout)

    def forward(self, x, xlen):
        x = x[:, :, :xlen.max()]
        _, _, x_utt = self.front(audio_signal=x, length=xlen) # x_utt: BxTxD; x: BxDxT
        return self.src_feat(x_utt), self.tgt_feat(x_utt)
