import logging
from tqdm import tqdm
import numpy as np
import torch
import networkx as nx

from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.datasets.data_util import load_graph_classification_dataset
from graphmae.models import build_model


def graph_classification_evaluation(model, pooler, dataloader, num_classes, lr_f, weight_decay_f, max_epoch_f, device,
                                    mute=False):
    model.eval()
    x_list = []
    y_list = []
    with torch.no_grad():
        for i, batch_g in enumerate(dataloader):
            batch_g = batch_g.to(device)
            feat = batch_g.x
            labels = batch_g.y.cpu()
            out = model.embed(feat, batch_g.edge_index)
            if pooler == "mean":
                out = global_mean_pool(out, batch_g.batch)
            elif pooler == "max":
                out = global_max_pool(out, batch_g.batch)
            elif pooler == "sum":
                out = global_add_pool(out, batch_g.batch)
            else:
                raise NotImplementedError

            y_list.append(labels.numpy())
            x_list.append(out.cpu().numpy())
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    test_f1, test_std = evaluate_graph_embeddings_using_svm(x, y)
    return test_f1


def evaluate_graph_embeddings_using_svm(embeddings, labels):
    result = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(embeddings, labels):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)

        preds = clf.predict(x_test)
        f1 = f1_score(y_test, preds, average="micro")
        result.append(f1)
    test_f1 = np.mean(result)
    test_std = np.std(result)
    with open(file_name, "a") as file:
        file.write(f"#Test_f1: {test_f1:.4f}±{test_std:.4f}\n")

    return test_f1, test_std
def getPagerank(train_loader, device):
    batch_idx = 0
    pagerank_values_tensor = []
    for batch in train_loader:
        batch_info = batch.batch
        num_graphs = batch_info[len(batch_info) - 1] + 1
        num_nodes = batch_info.shape[0]
        num_edges = batch.edge_index.shape[1]
        graph_list = [] #
        for graph_idx in range(num_graphs):
            graph_mask = batch_info == graph_idx
            edge_mask = graph_mask[batch.edge_index[0]]
            edge_indices = batch.edge_index[:, edge_mask]
            subgraph = edge_indices.clone().detach()
            g = nx.Graph()
            g.add_edges_from(subgraph.t().tolist())
            graph_list.append(g)

        pagerank_values = []
        for graph_idx in range(num_graphs):
            graph = graph_list[graph_idx]
            pr = nx.pagerank(graph)
            graph_mask = batch_info == graph_idx
            pagerank_values.append(np.array([pr[node] for node in range(num_nodes) if graph_mask[node]]))
        pagerank_values_new = []
        for arr in pagerank_values:
            pagerank_values_new.extend(arr)
        pagerank_values_tensor.append(torch.tensor(pagerank_values_new).to(device))
        batch_idx += 1
    return pagerank_values_tensor

def pretrain(model, pooler, dataloaders, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob=True, logger=None):
    global acc_list
    train_loader, eval_loader = dataloaders
    epoch_iter = tqdm(range(max_epoch))
    
    # 计算pagerank
    prs = getPagerank(train_loader, device)
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        batch_idx = 0

        for batch in train_loader:
            batch_g = batch
            batch_g = batch_g.to(device)
            feat = batch_g.x
            batch_info = batch_g.batch # 这个batch的batch信息，记录各个点分别属于哪个图
            # print(batch_info,end=" ") # 调试用
            model.train()
            loss, loss_dict = model(feat, batch_g.edge_index, batch_info, prs[batch_idx], epoch, max_epoch, args.curMode)
            batch_idx += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            if logger is not None:
                loss_dict["lr"] = get_current_lr(optimizer)
                logger.note(loss_dict, step=epoch)
        if scheduler is not None:
            scheduler.step()
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")
    return model


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler
    pooler = args.pooling
    deg4feat = args.deg4feat
    batch_size = args.batch_size

    graphs, (num_features, num_classes) = load_graph_classification_dataset(dataset_name, deg4feat=deg4feat)
    args.num_features = num_features

    train_idx = torch.arange(len(graphs))
    train_loader = DataLoader(graphs, batch_size=batch_size, pin_memory=True)
    eval_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(
                name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
            # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        if not load_model:
            model = pretrain(model, pooler, (train_loader, eval_loader), optimizer, max_epoch, device, scheduler,
                             num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        model = model.to(device)
        model.eval()
        test_f1 = graph_classification_evaluation(model, pooler, eval_loader, num_classes, lr_f, weight_decay_f, max_epoch_f, device, mute=False)
        acc_list.append(test_f1)
    with open(file_name, "a") as file:
        file.write(f"# final_acc: {np.mean(acc_list):.4f}±{np.std(acc_list):.4f}\n")


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    file_name = f"predefine_{args.dataset}.txt"
    acc_list = []
    main(args)