import argparse
from functools import partial

from loader import MoleculeDataset
from dataloader import DataLoaderMasking, DataLoaderMaskingPred #, DataListLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNNDecoder, MLP
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from util import MaskAtom

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from tensorboardX import SummaryWriter

from torch_geometric.nn.pool.topk_pool import topk


import timeit

import time

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)




def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def train_mae(args, model_list, loader, optimizer_list, device, epoch, alpha_l=1.0, loss_fn="sce"):
    torch.autograd.set_detect_anomaly(True)
    if loss_fn == "sce":
        criterion = partial(sce_loss, alpha=alpha_l)
    else:
        criterion = nn.CrossEntropyLoss()

    model, dec_pred_atoms, dec_pred_bonds, gnn_generator, mlp_generator = model_list
    optimizer_model, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds, optimizer_gnn_generator, optimizer_mlp_generator = optimizer_list

    # 开训
    model.train()
    dec_pred_atoms.train()
    gnn_generator.train()
    mlp_generator.train()

    if dec_pred_bonds is not None:
        dec_pred_bonds.train()

    loss_accum = 0
    acc_node_accum = 0
    acc_edge_accum = 0

    # 一些固定属性
    num_atom_type = 119
    num_edge_type = 5
    num_chirality_tag = 3
    num_bond_direction = 3

    epoch_iter = tqdm(loader, desc="Iteration")
    for step, oribatch in enumerate(epoch_iter):
        oribatch = oribatch.to(device)
        batch = oribatch.clone() # 克隆后的batch
        score_gnn_rep = gnn_generator(oribatch.x, oribatch.edge_index, oribatch.edge_attr)
        score_mlp_rep = mlp_generator(oribatch.x)
        scores = score_gnn_rep + args.sc_alpha_mlp * score_mlp_rep
        # scores = score_gnn_rep
        # 节点特征是多维的，要转化成分数，需要进行pooling操作
        if args.sc_pooling == 'mean':
            scores = scores.mean(dim = 1)
        elif args.sc_pooling == "sum":
            scores = scores.sum(dim = 1)
        elif args.sc_pooling == 'max':
            scores = scores.max(dim = 1)
        else:
            raise ValueError("Wrong sc_pooling")
        # 归一化
        scores = torch.sigmoid(scores).to(device)

        num_nodes = batch.x.size()[0]
        tmp_mask_rate = args.mask_rate * np.sqrt(epoch / args.epochs)
        throw_nodes_first = topk(scores, tmp_mask_rate, batch.batch).to(device)  # 选取得分最高的关注节点
        tmp_scores = torch.rand(num_nodes, device=device)
        tmp_scores[throw_nodes_first] += args.alpha  # 将关注节点的分数加上alpha
        masked_atom_indices = topk(tmp_scores, args.mask_rate, batch.batch).to(device) # 整个batch中应该被mask的节点
        all_indices = torch.arange(num_nodes).to(device)
        keep_nodes = all_indices[~torch.isin(all_indices, masked_atom_indices)].to(device)
        if(len(masked_atom_indices) == 0): # 没有需要mask的节点
            print("No node to mask!")
            continue

        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(batch.x[atom_idx.item()].view(1, -1))
        mask_node_label = torch.cat(mask_node_labels_list, dim=0)

        # ----------- graphMAE -----------
        atom_type = F.one_hot(mask_node_label[:, 0], num_classes=num_atom_type).float()
        node_attr_label = atom_type

        batch.x[masked_atom_indices,:] = torch.tensor([num_atom_type, 0]).to(device)

        time_mask_nodes = time.time()
        # print(f"mask节点特征耗时：{time_mask_nodes - time_onehot:.4f}秒")
        if args.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(batch.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                        bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]: # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        batch.edge_attr[bond_idx].view(1, -1))

                mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    batch.edge_attr[bond_idx] = torch.tensor(
                        [num_edge_type, 0])

                connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])

            else:
                mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

            edge_type = F.one_hot(mask_edge_label[:, 0], num_classes=num_edge_type).float()
            bond_direction = F.one_hot(mask_edge_label[:, 1], num_classes=num_bond_direction).float()
            edge_attr_label = torch.cat((edge_type, bond_direction), dim=1)
        node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
        node_rep_clone = node_rep.clone() # 避免原地操作
        node_rep_clone[keep_nodes] *= scores.view(scores.shape[0], -1)[keep_nodes]
        masked_node_indices = masked_atom_indices
        pred_node = dec_pred_atoms(node_rep_clone, batch.edge_index, batch.edge_attr, masked_node_indices)

        if loss_fn == "sce":
            loss = criterion(node_attr_label, pred_node[masked_node_indices])
        else:
            loss = criterion(pred_node.double()[masked_node_indices], mask_node_label[:,0])


        if args.mask_edge:
            masked_edge_index = batch.edge_index[:, connected_edge_indices]
            # edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
            edge_rep = node_rep_clone[masked_edge_index[0]] + node_rep_clone[masked_edge_index[1]]
            pred_edge = dec_pred_bonds(edge_rep)
            loss += criterion(pred_edge.double(), mask_edge_label[:,0])

            # acc_edge = compute_accuracy(pred_edge, batch.mask_edge_label[:,0])
            # acc_edge_accum += acc_edge

        optimizer_model.zero_grad()
        optimizer_dec_pred_atoms.zero_grad()
        # add scoreGenerator
        optimizer_gnn_generator.zero_grad()
        optimizer_mlp_generator.zero_grad()

        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.zero_grad()

        loss.backward()

        optimizer_model.step()
        optimizer_dec_pred_atoms.step()
        # add scoreGenerator
        optimizer_gnn_generator.step()
        optimizer_mlp_generator.step()

        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.step()

        loss_accum += float(loss.cpu().item())
        epoch_iter.set_description(f"train_loss: {loss.item():.4f}")

    return loss_accum/step #, acc_node_accum/step, acc_edge_accum/step



def main():
    # region Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.5,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--mask_edge', type=int, default=0,
                        help='whether to mask edges or not together with atoms')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset for pretraining')
    parser.add_argument('--output_model_file', type=str, default = '', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--input_model_file', type=str, default=None)
    parser.add_argument("--alpha_l", type=float, default=1.0)
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--decoder", type=str, default="gin")
    parser.add_argument("--use_scheduler", action="store_true", default=False)

    # add scoreGenerator
    parser.add_argument("--sc_num_layers_mlp", type=int, default=2)
    parser.add_argument("--sc_alpha_mlp", type = float, default=1.0)
    parser.add_argument("--sc_type", type = str, default="gin")
    parser.add_argument("--alpha", type = float, default=0.5)
    parser.add_argument("--sc_pooling", type = str, default = "mean")

    args = parser.parse_args()
    print(args)
    # endregion
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print("num layer: %d mask rate: %f mask edge: %d alpha: %f sc_alpha_mlp: %f" %(args.num_layer, args.mask_rate, args.mask_edge, args.alpha, args.sc_alpha_mlp))


    dataset_name = args.dataset
    # set up dataset and transform function.
    # dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset, transform = MaskAtom(num_atom_type = 119, num_edge_type = 5, mask_rate = args.mask_rate, mask_edge=args.mask_edge))
    dataset = MoleculeDataset("dataset/" + dataset_name, dataset=dataset_name)

    # loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    loader = DataLoaderMaskingPred(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, mask_rate=args.mask_rate, mask_edge=args.mask_edge)

    # set up models, one for pre-training and one for context embeddings
    model = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type).to(device)
    # scoreGenerator：1层GNN
    gnnGenerator = GNN(1, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type = args.gnn_type).to(device)
    # scoreGenerator：多层MLP
    mlpGenerator = MLP(args.sc_num_layers_mlp, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio).to(device)

    # linear_pred_atoms = torch.nn.Linear(args.emb_dim, 119).to(device)
    # linear_pred_bonds = torch.nn.Linear(args.emb_dim, 4).to(device)
    if args.input_model_file is not None and args.input_model_file != "":
        model.load_state_dict(torch.load(args.input_model_file))
        print("Resume training from:", args.input_model_file)
        resume = True
    else:
        resume = False

    NUM_NODE_ATTR = 119 # + 3 
    atom_pred_decoder = GNNDecoder(args.emb_dim, NUM_NODE_ATTR, JK=args.JK, gnn_type=args.gnn_type).to(device)
    if args.mask_edge:
        NUM_BOND_ATTR = 5 + 3
        bond_pred_decoder = GNNDecoder(args.emb_dim, NUM_BOND_ATTR, JK=args.JK, gnn_type=args.gnn_type)
        optimizer_dec_pred_bonds = optim.Adam(bond_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    else:
        bond_pred_decoder = None
        optimizer_dec_pred_bonds = None

    model_list = [model, atom_pred_decoder, bond_pred_decoder, gnnGenerator, mlpGenerator]

    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dec_pred_atoms = optim.Adam(atom_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_gnnGenerator = optim.Adam(gnnGenerator.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_mlpGenerator = optim.Adam(mlpGenerator.parameters(), lr=args.lr, weight_decay=args.decay)

    if args.use_scheduler:
        print("--------- Use scheduler -----------")
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / args.epochs) ) * 0.5
        scheduler_model = torch.optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda=scheduler)
        scheduler_dec = torch.optim.lr_scheduler.LambdaLR(optimizer_dec_pred_atoms, lr_lambda=scheduler)
        scheduler_gnnGenerator = torch.optim.lr_scheduler.LambdaLR(optimizer_gnnGenerator, lr_lambda=scheduler)
        scheduler_mlpGenerator = torch.optim.lr_scheduler.LambdaLR(optimizer_mlpGenerator, lr_lambda=scheduler)
        scheduler_list = [scheduler_model, scheduler_dec, scheduler_gnnGenerator, scheduler_mlpGenerator]
    else:
        scheduler_model = None
        scheduler_dec = None
        scheduler_gnnGenerator = None
        scheduler_mlpGenerator = None

    optimizer_list = [optimizer_model, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds, optimizer_gnnGenerator, optimizer_mlpGenerator]

    output_file_temp = "./checkpoints/" + "learnable" + f"_{args.gnn_type}" + f"_{args.mask_rate}" + f"_{args.alpha}" + f"_{args.sc_alpha_mlp}" + f"_{args.sc_num_layers_mlp}"

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        # train_loss, train_acc_atom, train_acc_bond = train(args, model_list, loader, optimizer_list, device)
        # print(train_loss, train_acc_atom, train_acc_bond)

        train_loss = train_mae(args, model_list, loader, optimizer_list, device, epoch, alpha_l=args.alpha_l, loss_fn=args.loss_fn)
        if not resume:
            if epoch % 50 == 0:
                torch.save(model.state_dict(), output_file_temp + f"_{epoch}.pth")
        print(train_loss)
        if scheduler_model is not None:
            scheduler_model.step()
        if scheduler_dec is not None:
            scheduler_dec.step()
        if scheduler_gnnGenerator is not None:
            scheduler_gnnGenerator.step()
        if scheduler_mlpGenerator is not None:
            scheduler_mlpGenerator.step()

    output_file = "./checkpoints/" + args.output_model_file + f"_{args.gnn_type}"
    if resume:
        torch.save(model.state_dict(), args.input_model_file.rsplit(".", 1)[0] + f"_resume_{args.epochs}.pth")
    elif not args.output_model_file == "":
        torch.save(model.state_dict(), output_file + ".pth")

if __name__ == "__main__":
    main()
