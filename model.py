import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import scatter
from torch_geometric.nn import SAGEConv
from torch.utils.checkpoint import checkpoint
from torch.nn import Linear, Parameter
from typing import Union, Tuple, Optional
from pytorch_lightning import LightningModule
import numpy as np

# ----------------------
# Utility functions
# ----------------------
def init_linear(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def t2np(x):
    return x.detach().cpu().numpy()

# ----------------------
# Custom SAGEConvV2
# ----------------------
class SAGEConvV2(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int,int]], out_channels:int, 
                 normalize:bool=False, root_weight=True, bias=True, aggr='mean'):
        super().__init__(aggr=aggr)
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)
        self.lin_l = nn.Sequential(
            nn.Linear(in_channels[0]*2, in_channels[0]*2, bias=bias),
            nn.ReLU(),
            nn.Linear(in_channels[0]*2, out_channels, bias=bias)
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.apply(init_linear)
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], edge_index, edge_attr=None):
        if isinstance(x, torch.Tensor):
            x = (x, x)
        out = self.propagate(edge_index, x=x)
        if self.root_weight and x[1] is not None:
            out += self.lin_r(x[1])
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def message(self, x_i, x_j):
        h = torch.cat([x_i, x_j], dim=-1)
        return self.lin_l(h)

# ----------------------
# DeepGCNLayerV2
# ----------------------
class DeepGCNLayerV2(nn.Module):
    def __init__(self, conv=None, norm=None, act=None, block='res+', dropout=0., ckpt_grad=False):
        super().__init__()
        self.conv = conv
        self.norm = norm
        self.act = act
        self.block = block.lower()
        self.dropout = dropout
        self.ckpt_grad = ckpt_grad

    def forward(self, x, edge_index, edge_attr=None):
        h = x
        if self.norm: h = self.norm(h)
        if self.act: h = self.act(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        if self.conv:
            if self.ckpt_grad and h.requires_grad:
                h = checkpoint(self.conv, h, edge_index)
            else:
                h = self.conv(h, edge_index)
        if self.block in ['res','res+']:
            h = x + h
        return h

# ----------------------
# DeeperGCN
# ----------------------
class DeeperGCN(nn.Module):
    def __init__(self, in_channel, mid_channel, num_layers, graph_conv=SAGEConvV2, node_encoding=True, dropout_ratio=0.1):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channel, mid_channel),
            nn.LayerNorm(mid_channel)
        ) if node_encoding else None

        self.gcn_blocks = nn.ModuleList()
        for _ in range(num_layers):
            conv = graph_conv(mid_channel, mid_channel)
            norm = nn.LayerNorm(mid_channel)
            act = nn.ReLU(inplace=True)
            self.gcn_blocks.append(DeepGCNLayerV2(conv, norm, act, block='res+', dropout=dropout_ratio))

    def forward(self, x, edge_index):
        if self.node_encoder: x = self.node_encoder(x)
        for layer in self.gcn_blocks:
            x = layer(x, edge_index)
        return x

# ----------------------
# CNN module
# ----------------------
class CNN(nn.Module):
    def __init__(self, in_channel, mid_channel, seq_len, filters=[32,64,96], kernels=[4,6,8], dropout_ratio=0.1):
        super().__init__()
        self.conv = nn.ModuleList()
        in_ch = [in_channel] + filters
        for i in range(len(filters)):
            self.conv.append(nn.Conv1d(in_ch[i], in_ch[i+1], kernel_size=kernels[i]))
        self.fc1 = nn.Linear(filters[-1], mid_channel)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        for l in self.conv:
            x = F.relu(l(x))
        x = F.adaptive_max_pool1d(x,1).squeeze(-1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

# ----------------------
# DeepDrug model
# ----------------------
class DeepDrug(nn.Module):
    def __init__(self, in_channel=93, mid_channel=128, num_out_dim=1, seq_len=200, dropout_ratio=0.1):
        super().__init__()
        self.gconv = DeeperGCN(in_channel, mid_channel, num_layers=4)
        self.gconv_seq = CNN(len(smile_dict), mid_channel, seq_len)
        self.global_fc_nn = nn.Sequential(
            nn.Linear(mid_channel*2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, num_out_dim)
        )

    def forward(self, graph_data, seq_data):
        x, edge_index = graph_data.x, graph_data.edge_index
        h1 = self.gconv(x, edge_index)
        h1 = global_mean_pool(h1, graph_data.batch)
        h2 = self.gconv_seq(seq_data)
        h = torch.cat([h1, h2], dim=-1)
        out = self.global_fc_nn(h)
        return out

# ----------------------
# Lightning container
# ----------------------
class DeepDrug_Container(LightningModule):
    def __init__(self, num_out_dim=1, lr=1e-3, task_type='regression'):
        super().__init__()
        self.save_hyperparameters()
        self.model = DeepDrug(num_out_dim=num_out_dim)
        if task_type=='regression':
            self.loss_func = F.mse_loss
        elif task_type in ['binary','binary_classification']:
            self.loss_func = F.binary_cross_entropy_with_logits
        else:
            self.loss_func = F.cross_entropy
        self.task_type = task_type

    def forward(self, batch):
        graph_data, seq_data, y = batch
        return self.model(graph_data, seq_data)

    def training_step(self, batch, batch_idx):
        graph_data, seq_data, y = batch
        y_hat = self.model(graph_data, seq_data)
        loss = self.loss_func(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        graph_data, seq_data, y = batch
        y_hat = self.model(graph_data, seq_data)
        loss = self.loss_func(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }
