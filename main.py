import logging
from webbrowser import get
from matplotlib import test
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.datasets.data_util import load_dataset
from graphmae.evaluation import node_classification_evaluation,  node_classification, node_classification_eval
from graphmae.models import build_model
import pandas as pd 
import networkx as nx
import dgl
import torch
from sklearn import model_selection
import numpy as np
import pickle

def get_ppi():
    '''high_conf_edgelist_1 = pd.read_csv('/data/guest/GraphMAE/graphmae/datasets/ppi.csv', sep='\t').astype(str)
    G = nx.from_pandas_edgelist(df=high_conf_edgelist_1, source='partner1', target='partner2', edge_attr='confidence')
    k = pd.read_csv('/data/guest/GraphMAE/graphmae/datasets/ncg_label.csv', sep='\t').astype(str)
    positive = k['entrez'].unique()
    for node in list(G.nodes):
        if node in positive:
            G.nodes[node]['label'] = 1
        else:
            G.nodes[node]['label'] = 0
    x = dgl.from_networkx(G, node_attrs=['label'])
    x.ndata['feat'] = torch.nn.functional.one_hot(x.in_degrees(x.nodes()))
    train_mask, test_mask, _, _= model_selection.train_test_split(np.arange(x.num_nodes()), np.arange(x.num_nodes()), random_state=33)
    mask = torch.zeros(x.num_nodes()).long()
    mask[train_mask] = 1
    x.ndata['train_mask'] = mask
    mask = torch.zeros(x.num_nodes()).long()
    mask[test_mask] = 1
    x.ndata['test_mask'] = mask
    x.ndata['val_mask'] = mask'''
    data = torch.load("/data/guest/MTGCN/data/CPDB_data.pkl")
    data = Data(**data.__dict__).to("cpu")
    datas = torch.load("/data/guest/MTGCN/data/str_fearures.pkl").to("cpu")
    data.x = data.x[:, :48]
    #data.x = torch.cat((data.x, datas), 1)
    x = dgl.graph(((data.edge_index[0]), (data.edge_index[1])))
    x = dgl.to_bidirected(x)
    x = dgl.remove_self_loop(x)
    x = dgl.add_self_loop(x)
    x.ndata['label'] = torch.from_numpy(np.logical_or(data.y, data.y_te)).float()
    x.ndata['feat'] = data.x
    x.ndata['train_mask'] = torch.from_numpy(data.mask)
    x.ndata['test_mask'] = torch.from_numpy(data.mask_te)
    x.ndata['val_mask'] = torch.from_numpy(data.mask_te)
    return x, (x.ndata["feat"].shape[1], 2)


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()

        loss, loss_dict = model(graph, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        #if (epoch + 1) % 5 == 0:
            #node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=True, mute=True)
    # return best_model
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

    graph, (num_features, num_classes) = get_ppi()   
    args.num_features = num_features

    acc_list = []
    estp_acc_list = []
  

    
    with open("/data/guest/MTGCN/data/k_sets.pkl", 'rb') as handle:
        k_sets = pickle.load(handle)

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
        
        print(load_model, save_model)
            
        x = graph.ndata["feat"].float()
        if not load_model:
            model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")
        
        model = model.to(device)
        model.eval()

        final_acc, estp_acc = node_classification_eval(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=False)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

        if logger is not None:
            logger.finish()

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")


import sys

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    torch.cuda.set_device(2)
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    try:
        main(args)
    except Exception:
        import traceback
        import pdb

        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
 