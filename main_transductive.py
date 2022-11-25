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
    mask_edge,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.datasets.data_util import load_dataset
from graphmae.evaluation import node_classification_evaluation,  node_classification
from graphmae.models import build_model
import pandas as pd 
import networkx as nx
import dgl
import torch
from sklearn import model_selection
import numpy as np
import pickle
import h5py


Dataset = ['CPDB', 'IRefIndex', 'PCNet', 'IRefIndex_2015', 'STRINGdb', 'Multinet']

def get_health(dataset=0):
    k = pd.read_csv('/data/guest/GraphMAE/graphmae/datasets/health.tsv', sep='\t').astype(str)
    f = h5py.File('/data/guest/GraphMAE/graphmae/datasets/EMOGI_{}/{}_multiomics.h5'.format(dataset, dataset), 'r') 
    positive = k['symbol'].unique()
    name = [x.decode() for x in list(f['gene_names'][:][..., 1])]
    mask_test = f['mask_test'][:]
    mask_train = f['mask_train'][:]
    y_train = f['y_train'][:]
    positive_mask_test = np.nonzero(mask_test == True)[0]
    positive_mask_train = np.nonzero(mask_test == True)[0]
    neg_mask_train = []
    neg_mask_test = []
    for gene in positive[:60]:
        try: 
            i = name.index(gene)
            neg_mask_train.append(i)
        except ValueError:
            pass
    for gene in positive[60:]:
        try: 
            i = name.index(gene)
            neg_mask_test.append(i)
        except ValueError:
            pass
    pos_mask_test = positive_mask_test[:60]
    pos_mask_train = positive_mask_train[:120]
    t_mask = list(set(pos_mask_train) | set(neg_mask_train))
    train_mask = np.zeros_like(mask_train)
    test_mask = np.zeros_like(mask_test)
    te_mask = list(set(pos_mask_test) | set(neg_mask_test))
    train_mask[t_mask] = True
    test_mask[te_mask] = True
    print(len(t_mask), len(te_mask))
    y = np.zeros_like(y_train)
    for i in neg_mask_train:
        y[i] = True
    for i in neg_mask_test:
        y[i] = True
    src, dst = np.nonzero(f['network'][:])
    graph = dgl.graph((src, dst))
    graph.ndata['feat'] = torch.from_numpy(f['features'][:])   
    graph.ndata['train_mask'] = torch.from_numpy(train_mask)
    graph.ndata['val_mask'] = torch.from_numpy(test_mask)
    graph.ndata['test_mask'] = torch.from_numpy(test_mask)
    graph.ndata['label'] = torch.from_numpy(y).float()
    return graph, (graph.ndata["feat"].shape[1], 2)


def get_ppi(dataset=0, essential_gene=False, health_gene=False):
    dataset = Dataset[dataset]
    if health_gene:
        return get_health(dataset)
    elif essential_gene:
        f = h5py.File('/data/guest/GraphMAE/graphmae/essential_gene/{}_essential_test01_multiomics.h5'.format(dataset), 'r')
    else:
        f = h5py.File('/data/guest/GraphMAE/graphmae/datasets/EMOGI_{}/{}_multiomics.h5'.format(dataset, dataset), 'r') 
    src, dst = np.nonzero(f['network'][:])
    graph = dgl.graph((src, dst))
    #datas = torch.load("/data/guest/MTGCN/data/str_fearures.pkl").to("cpu")
    graph.ndata['feat'] = torch.from_numpy(f['features'][:])
    graph.ndata['train_mask'] = torch.from_numpy(f['mask_train'][:])
    graph.ndata['val_mask'] = torch.from_numpy(f['mask_test'][:])
    graph.ndata['test_mask'] = torch.from_numpy(f['mask_test'][:])
    graph.ndata['label'] = torch.from_numpy(np.logical_or(np.logical_or(f['y_test'][:], f['y_val'][:]), f['y_train'][:])).float()
    print(graph.ndata["feat"].shape[1])
    return graph, (graph.ndata["feat"].shape[1], 2)



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

    graph, (num_features, num_classes) = get_ppi(dataset=args.ppi, essential_gene=args.expression, health_gene=args.health)   
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

        if use_scheduler and max_epoch !=0:
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


        scheduler_f = None   
   
        
        inducive_dataset = args.inductive_ppi
        final_acc, estp_acc = node_classification_evaluation(model, inducive_dataset, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, scheduler_f, linear_prob=False)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

        if logger is not None:
            logger.finish()

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")
    np.savetxt('gcn_{}'.format(args.weight_decay_f), np.array([estp_acc]))


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
 