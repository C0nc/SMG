import logging
from tqdm import tqdm
import numpy as np
import random

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl.dataloading import GraphDataLoader

import torch
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)

import sys 
from graphmae.datasets.data_util import load_graph_classification_dataset
from graphmae.models import build_model
from GNNSubNet.GNNSubNet.dataset import load_OMICS_dataset, convert_to_s2vgraph


def get_graph():
    loc   = "/home/yancui/ppimae/GNNSubNet/TCGA"
# PPI network
    ppi   = f'{loc}/KIDNEY_RANDOM_PPI.txt'
# single-omic features
#feats = [f'{loc}/KIDNEY_RANDOM_Methy_FEATURES.txt']
# multi-omic features
    feats = [f'{loc}/KIDNEY_RANDOM_mRNA_FEATURES.txt', f'{loc}/KIDNEY_RANDOM_Methy_FEATURES.txt']
# outcome class
    targ  = f'{loc}/KIDNEY_RANDOM_TARGET.txt'
    dataset, gene_names =load_OMICS_dataset(ppi, feats, targ, True, 950, True)
    graphs_class_0_list = []
    graphs_class_1_list = []
    for graph in dataset:
        if graph.y.numpy() == 0:
            graphs_class_0_list.append(graph)
        else:
            graphs_class_1_list.append(graph)

    graphs_class_0_len = len(graphs_class_0_list)
    graphs_class_1_len = len(graphs_class_1_list)

    print(f"Graphs class 0: {graphs_class_0_len}, Graphs class 1: {graphs_class_1_len}")

    ########################################################################################################################
    # Downsampling of the class that contains more elements ===========================================================
    # ########################################################################################################################

    if graphs_class_0_len >= graphs_class_1_len:
        random_graphs_class_0_list = random.sample(graphs_class_0_list, graphs_class_1_len)
        balanced_dataset_list = graphs_class_1_list + random_graphs_class_0_list

    if graphs_class_0_len < graphs_class_1_len:
        random_graphs_class_1_list = random.sample(graphs_class_1_list, graphs_class_0_len)
        balanced_dataset_list = graphs_class_0_list + random_graphs_class_1_list

    #print(len(random_graphs_class_0_list))
    #print(len(random_graphs_class_1_list))

    random.shuffle(balanced_dataset_list)
    print(f"Length of balanced dataset list: {len(balanced_dataset_list)}")

    list_len = len(balanced_dataset_list)
    #print(list_len)
    train_set_len = int(list_len * 4 / 5)
    train_dataset_list = balanced_dataset_list[:train_set_len]
    test_dataset_list  = balanced_dataset_list[train_set_len:]

    train_graph_class_0_nr = 0
    train_graph_class_1_nr = 0
    for graph in train_dataset_list:
        if graph.y.numpy() == 0:
            train_graph_class_0_nr += 1
        else:
            train_graph_class_1_nr += 1
    print(f"Train graph class 0: {train_graph_class_0_nr}, train graph class 1: {train_graph_class_1_nr}")

    test_graph_class_0_nr = 0
    test_graph_class_1_nr = 0
    for graph in test_dataset_list:
        if graph.y.numpy() == 0:
            test_graph_class_0_nr += 1
        else:
            test_graph_class_1_nr += 1
    print(f"Validation graph class 0: {test_graph_class_0_nr}, validation graph class 1: {test_graph_class_1_nr}")

    s2v_train_dataset = convert_to_s2vgraph(train_dataset_list)
    s2v_test_dataset  = convert_to_s2vgraph(test_dataset_list)
    return s2v_train_dataset, s2v_test_dataset


def graph_classification_evaluation(model, pooler, dataloader,  num_classes, lr_f, weight_decay_f, max_epoch_f, device, mute=False):
    model.eval()
    x_list = []
    y_list = []
    with torch.no_grad():
        for i, (batch_g, labels) in enumerate(dataloader):
            batch_g = batch_g.to(device)
            feat = batch_g.ndata["attr"]
            out = model.embed(batch_g, feat)
            out = pooler(batch_g, out)

            y_list.append(labels.numpy())
            x_list.append(out.cpu().numpy())
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    test_f1, test_std = evaluate_graph_embeddings_using_svm(x, y)
    print(f"#Test_f1: {test_f1:.4f}±{test_std:.4f}")
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
        acc = accuracy_score(y_test, preds)
        result.append(acc)
    test_f1 = np.mean(result)
    test_std = np.std(result)

    return test_f1, test_std


def pretrain(model, pooler, dataloaders, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob=True, logger=None):
    train_loader, eval_loader = dataloaders

    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        for batch in train_loader:
            batch_g, _ = batch
            batch_g = batch_g.to(device)

            feat = batch_g.ndata["attr"]
            model.train()
            loss, loss_dict = model(batch_g, feat)
            
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

            
def collate_fn(batch):
    # graphs = [x[0].add_self_loop() for x in batch]
    graphs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    batch_g = dgl.batch(graphs)
    labels = torch.cat(labels, dim=0)
    return batch_g, labels


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
    pooling = args.pooling
    deg4feat = args.deg4feat
    batch_size = args.batch_size

    train_set, test_set = get_graph()
    graphs = train_set + test_set
    num_features = graphs[0][0].ndata["attr"].shape[1]
    num_classes = 2
    
    args.num_features = num_features

    train_idx = torch.arange(len(graphs))
    train_sampler = SubsetRandomSampler(train_idx)
    
    train_loader = GraphDataLoader(graphs, sampler=train_sampler, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)
    eval_loader = GraphDataLoader(graphs, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

    if pooling == "mean":
        pooler = AvgPooling()
    elif pooling == "max":
        pooler = MaxPooling()
    elif pooling == "sum":
        pooler = SumPooling()
    else:
        raise NotImplementedError

    acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        if not load_model:
            model = pretrain(model, pooler, (train_loader, eval_loader), optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob,  logger)
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

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    args.max_epoch = 0
    print(args)
    main(args)
