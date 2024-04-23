from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor

from .gat import GAT
from .gin import GIN
from .gcn import GCN
from .mlp import MLP
from .loss_func import sce_loss
from graphmae.utils import create_norm
from torch_geometric.utils import dropout_edge
from torch_geometric.utils import add_self_loops, remove_self_loops
# from torch_geometric.nn.pool.topk_pool import topk

from typing import Callable, Optional, Tuple, Union
from torch.nn import Parameter

from torch_geometric.nn.inits import uniform
from torch_geometric.utils import scatter, softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes

import numpy as np


def topk(
        x: Tensor,
        ratio: Optional[Union[float, int]],
        batch: Tensor,
        min_score: Optional[float] = None,
        tol: float = 1e-7,
) -> Tensor:
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)

    elif ratio is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')
        batch_size, max_num_nodes = num_nodes.size(0), int(num_nodes.max())

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes,), -60000.0)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0),), int(ratio))
            k = torch.min(k, num_nodes)
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        if isinstance(ratio, int) and (k == ratio).all():
            # If all graphs have exactly `ratio` or more than `ratio` entries,
            # we can just pick the first entries in `perm` batch-wise:
            index = torch.arange(batch_size, device=x.device) * max_num_nodes
            index = index.view(-1, 1).repeat(1, ratio).view(-1)
            index += torch.arange(ratio, device=x.device).repeat(batch_size)
        else:
            # Otherwise, compute indices per graph:
            index = torch.cat([
                torch.arange(k[i], device=x.device) + i * max_num_nodes
                for i in range(batch_size)
            ], dim=0)

        perm = perm[index]

    else:
        raise ValueError("At least one of 'min_score' and 'ratio' parameters "
                         "must be specified")

    return perm

def getMaskRate(mask_rate, epoch, max_epoch, mode):
    lambda0 = 0.05
    if mode == "linear":
        tmp_mask_rate = mask_rate * epoch / max_epoch
    elif mode == "root":
        tmp_mask_rate = mask_rate * np.sqrt(epoch / max_epoch)
    elif mode == "geometric": # fix geometric error
        tmp_mask_rate = mask_rate * np.power(2, np.log2(lambda0) - np.log2(lambda0) * epoch / max_epoch)
    elif mode == "None":
        tmp_mask_rate = mask_rate
    else:
        raise Exception("curMode error!")
    # print(f"{mode}  {epoch} in {max_epoch}: {tmp_mask_rate}")
    return tmp_mask_rate

def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead,
                 nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=int(in_dim),
            num_hidden=int(num_hidden),
            out_dim=int(out_dim),
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "mlp":
        mod = MLP(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    else:
        raise NotImplementedError

    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gin",
            decoder_type: str = "gin",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            sc_type: str = "gin",
            sc_type2: str = "mlp",
            sc_num_layers1: int = 1,  # gnn层数
            sc_num_layers2: int = 1,  # mlp层数
            alpha: float = 0.1,
            alpha_sc2: float = 0.0,
            sc_sigmoid: str = "True",
    ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden

        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self._alpha = alpha
        self._alpha_sc2 = alpha_sc2
        self.sc_sigmoid =sc_sigmoid

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0

        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        # 第一个scoreGenerator
        if sc_type in ("gat", "dotgat"):
            sc_num_hidden = num_hidden // nhead
            sc_nhead = nhead
        else:
            sc_num_hidden = num_hidden
            sc_nhead = 1

        # 第二个scoreGenerator
        if sc_type2 in ("gat", "dotgat"):
            sc_num_hidden2 = num_hidden // nhead
            sc_nhead2 = nhead
        else:
            sc_num_hidden2 = num_hidden
            sc_nhead2 = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden

        # 这一层用于计算分数
        self.scoreGenrator = setup_module(
            m_type=sc_type,
            enc_dec="encoding",
            in_dim=in_dim,  # 输入特征维度
            num_hidden=sc_num_hidden,
            out_dim=1,  # 输出特征维度为1，因为是计算分数
            num_layers=sc_num_layers1,
            nhead=sc_nhead,
            nhead_out=sc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )
        # another scoreGenerator
        self.scoreGenrator2 = setup_module(
            m_type=sc_type2,
            enc_dec="encoding",
            in_dim=in_dim,  # 输入特征维度
            num_hidden=sc_num_hidden2,
            out_dim=1,  # 输出特征维度为1，因为是计算分数
            num_layers=sc_num_layers2,
            nhead=sc_nhead2,
            nhead_out=sc_nhead2,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )
        # endregion
        # region build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )
        # endregion
        # region build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )
        # endregion
        # region other layers
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        # endregion
        # region setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        # endregion

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, x, mask_rate, scores, batch, epoch, max_epoch, curMode, start_learning_epoch):
        num_nodes = x.shape[0]
        if epoch >= start_learning_epoch:
            tmp_mask_rate = getMaskRate(mask_rate, epoch, max_epoch, curMode)
            throw_nodes_first = topk(scores, tmp_mask_rate, batch).to(x.device)  # 选取得分最高的关注节点
            tmp_scores = torch.tensor(np.random.uniform(0, 1, num_nodes)).to(x.device)  # 生成随机分数
            tmp_scores[throw_nodes_first] += self._alpha  # 将关注节点的分数加上alpha
            keep_nodes = topk(-tmp_scores, 1 - mask_rate, batch).to(x.device)  # 保留得分最低的1 - mask_rate的节点
            all_indices = torch.arange(num_nodes).to(x.device)
            mask_nodes = all_indices[~torch.isin(all_indices, keep_nodes)].to(x.device)
        else:
            perm = torch.randperm(num_nodes, device=x.device)
            num_mask_nodes = int(mask_rate * num_nodes)
            mask_nodes = perm[: num_mask_nodes]
            keep_nodes = perm[num_mask_nodes: ]

        out_x = x.clone()
        if self._replace_rate > 0:
            num_mask_nodes = len(mask_nodes)
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            # out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            # out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        # 乘上分数
        if epoch >= start_learning_epoch:
            out_x[keep_nodes] *= scores.view(scores.shape[0], 1)[keep_nodes]  # 特征乘上分数，使之有梯度

        return out_x, (mask_nodes, keep_nodes)

    def forward(self, x, edge_index, batch, epoch, max_epoch, curMode, start_learning_epoch):
        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(x, edge_index, batch, epoch, max_epoch, curMode, start_learning_epoch)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def mask_attr_prediction(self, x, edge_index, batch, epoch, max_epoch, curMode, start_learning_epoch):
        # 计算分数1
        scores_rep = self.scoreGenrator(x, edge_index)  # 计算分数
        # print(scores_rep.shape)
        scores1 = scores_rep.mean(dim=1)
        # scores1 = scores_rep

        # 计算分数2
        scores_rep2 = self.scoreGenrator2(x, edge_index)  # 计算分数
        # print(scores_rep.shape)
        scores2 = scores_rep2.mean(dim=1)
        # scores2 = scores_rep2

        # 两个分数加权融合
        scores = scores1 + self._alpha_sc2 * scores2

        if self.sc_sigmoid == "True":
            scores = torch.sigmoid(scores)
        else:
            min_score = torch.min(scores)
            max_score = torch.max(scores)
            scores = (scores - min_score) / (max_score - min_score)
        use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, self._mask_rate, scores, batch, epoch, max_epoch, curMode, start_learning_epoch)

        if self._drop_edge_rate > 0:
            use_edge_index, masked_edges = dropout_edge(edge_index, self._drop_edge_rate)
            use_edge_index = add_self_loops(use_edge_index)[0]
        else:
            use_edge_index = edge_index

        enc_rep, all_hidden = self.encoder(use_x, use_edge_index, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "linear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(rep, use_edge_index)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)
        return loss

    def embed(self, x, edge_index):
        rep = self.encoder(x, edge_index)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
