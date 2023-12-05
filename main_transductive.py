import logging
from webbrowser import get
#from matplotlib import test
import numpy as np
from tqdm import tqdm
import torch
from sklearn.model_selection import KFold

from graphmae.utils import (
    build_args,
    create_optimizer,
    mask_edge,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
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
#from openTSNE import TSNE

#import utils
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



def result(pred, true):
    aa = torch.sigmoid(pred)
    precision, recall, _thresholds = metrics.precision_recall_curve(true, aa)
    area = metrics.auc(recall, precision)
    return metrics.roc_auc_score(true, aa), area, precision, recall


#f = h5py.File(PATH+'{}_multiomics.h5'.format(Dataset[0]), 'r') 
#genes = f['gene_names'][...,-1].astype(str)

#np.save('genes', genes)

def draw_tSNE(embed, y, ppi):
    tsne = TSNE(
    perplexity=30,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
)
    
    embedding_train = tsne.fit(embed.detach().cpu().numpy())
    utils.plot(embedding_train, y.cpu().numpy().astype(int).squeeze(-1), colors=utils.MOUSE_10X_COLORS, ppi=ppi)
    return


Dataset = ['CPDB', 'IRefIndex', 'PCNet', 'IRefIndex_2015', 'STRINGdb', 'Multinet']

PATH = '/data/civy/SMG/smg_data/'



def get_health(dataset=0):
    f = h5py.File('/data/civy/SMG/smg_data/'+'{}_multiomics.h5'.format(dataset), 'r') 
    src, dst = np.nonzero(f['network'][:])
    graph = dgl.graph((src, dst))
    graph.ndata['name'] = torch.arange(f['features'][:].shape[0]).unsqueeze(1)
    graph.ndata['feat'] = torch.from_numpy(f['features'][:])

    gene_name = f['gene_names'][...,-1].astype(str)
    gene_map = {g:i for i, g in enumerate(gene_name)}
    label = np.logical_or(np.logical_or(f['y_test'][:], f['y_val'][:]), f['y_train'][:]).squeeze(-1)
    neg_label = gene_name[label]

    label = pd.read_csv(PATH + 'health.tsv', sep='\t').astype(str)['symbol'].tolist()

    mask = [gene_map[g] for g in sorted(list((set(label)) & set(gene_name)))]

    #neg_mask = [gene_map[g] for g in list(set(neg_label) - set(label))]
    np.random.seed(42)
    neg_mask = np.random.choice(sorted(list(set(np.arange(len(gene_name))) - set(mask))), min(len(mask),len(gene_name) - len(mask)) , replace=False).tolist()

    print(neg_mask)
    label = torch.zeros(len(gene_name), dtype=torch.float)

    label[mask] = 1

    labels = label.unsqueeze(1)
    
    graph.ndata['label'] = labels
    
    final_mask = mask + neg_mask

    train_mask, test_mask, _, _ = train_test_split(final_mask, label[final_mask].numpy(), test_size = 0.2, shuffle=True, stratify=label[final_mask].numpy(), random_state=42)
    train_mask, val_mask, _, _ = train_test_split(train_mask, label[train_mask].numpy(), test_size = 0.2, shuffle=True, stratify=label[train_mask].numpy(),random_state=42)
    
    index = torch.zeros(len(gene_name), dtype=torch.bool)
    index[train_mask] = 1
    
    graph.ndata['train_mask'] = index.unsqueeze(1)
    
    index = torch.zeros(len(gene_name), dtype=torch.bool)
    index[test_mask] = 1

    graph.ndata['test_mask'] = index.unsqueeze(1)
    
    index = torch.zeros(len(gene_name), dtype=torch.bool)
    index[val_mask] = 1

    graph.ndata['val_mask'] = index.unsqueeze(1)
    
    graph = dgl.add_self_loop(graph)
     

    return graph, (graph.ndata["feat"].shape[1], 1)


def get_ppi(dataset=0, essential_gene=False, health_gene=False):
    dataset = Dataset[dataset]
    if health_gene:
        return get_health(dataset)
    elif essential_gene:
         f = h5py.File('/data/civy/SMG/smg_data/'+'{}_essential_test01_multiomics.h5'.format(dataset), 'r') 
    else:
        f = h5py.File('/data/civy/SMG/smg_data/'+'{}_multiomics.h5'.format(dataset), 'r')         
    src, dst = np.nonzero(f['network'][:])
    graph = dgl.graph((src, dst))
    graph.ndata['name'] = torch.arange(f['features'][:].shape[0]).unsqueeze(1)
    graph.ndata['feat'] = torch.from_numpy(f['features'][:])
    graph.ndata['train_mask'] = torch.from_numpy(f['mask_train'][:])
    graph.ndata['val_mask'] = torch.from_numpy(f['mask_val'][:])
    graph.ndata['test_mask'] = torch.from_numpy(f['mask_test'][:])
    full_mask = np.arange(graph.ndata['val_mask'].shape[0])[graph.ndata['test_mask'] | graph.ndata['train_mask']]
    graph.ndata['label'] = torch.from_numpy(np.logical_or(np.logical_or(f['y_test'][:], f['y_val'][:]), f['y_train'][:])).float()
    gene_name = f['gene_names'][...,-1].astype(str)
    gene_map = {g:i for i, g in enumerate(gene_name)}
    neg_label = pd.read_csv(PATH + 'health.tsv', sep='\t').astype(str)['symbol'].tolist()
    neg_test_mask = [gene_map[g] for g in sorted(list(set(neg_label) & set(gene_name[~(graph.ndata['train_mask'] | graph.ndata['val_mask'])])))]
    label_transfer = torch.zeros(len(gene_name), dtype=torch.float)

    label = np.logical_or(np.logical_or(f['y_test'][:], f['y_val'][:]), f['y_train'][:]).squeeze() 
    #print(gene_name[label])
    #print(len(list(set(neg_label) & set(gene_name[label]))), len(list(set(neg_label) & set(gene_name))))
    
    label_transfer[neg_test_mask] = 1
    
    y_test = graph.ndata['label'][graph.ndata['test_mask']].squeeze(-1)
    y_neg = np.arange(graph.ndata['test_mask'].shape[0])[graph.ndata['label'].squeeze(-1) == 0]
    
    transfer_mask = torch.arange(graph.ndata['test_mask'].shape[0])[graph.ndata['test_mask']][y_test==1].tolist() +  list(set(neg_test_mask) & set(y_neg))

    return graph, (graph.ndata["feat"].shape[1], 1)

def get_new_ppi_c(e=False, h=False):
    interactions = pd.read_csv('/data/civy/SMG/smg_data/ConsensusPathDB_human_PPI.gz',
                            compression='gzip',
                            header=1,
                            sep='\t',
                            encoding='utf8'
                            )
    interactions_nona = interactions.dropna()
    interactions_nona.head()
    # select interactions with exactly two partners
    binary_inter = interactions_nona[interactions_nona.interaction_participants__genename.str.count(',') == 1]
    # split the interactions columns into interaction partners
    edgelist = pd.concat([binary_inter.interaction_participants__genename.str.split(',', expand=True),
                                    binary_inter.interaction_confidence], axis=1
                                )
    # make the dataframe beautiful
    edgelist.set_index([np.arange(edgelist.shape[0])], inplace=True)
    edgelist.columns = ['partner1', 'partner2', 'confidence']

    high_conf_edgelist = edgelist[edgelist.confidence > .5]

    _, gene_list_1 = pd.factorize(high_conf_edgelist['partner1'])

    _, gene_list_2 = pd.factorize(high_conf_edgelist['partner2'])


    feature_list =  pd.read_csv('/data/civy/SMG/smg_data/multiomics_features.tsv', sep='\t')

    feature_name = feature_list['Unnamed: 0'].tolist() 

    gene_list = sorted(list((set(gene_list_1) | set(gene_list_2))))

    gene_map  = {g: i for i, g in enumerate(gene_list)}

    feature_map = {g: i for i, g in enumerate(feature_name)}

    src = [gene_map[g] for g in high_conf_edgelist['partner1'].tolist()]

    dst = [gene_map[g] for g in high_conf_edgelist['partner2'].tolist()]

    mask = [gene_map[g] for g in sorted(list(set(gene_list) & set(feature_name)))]

    feature = [feature_list.iloc[feature_map[g], 1:].tolist() for g in sorted(list(set(gene_list) & set(feature_name)))]


    feats =  torch.zeros((len(gene_map), 64))

    feats[mask] = torch.tensor(feature, dtype=torch.float)
    

    graph = dgl.graph((src, dst))

    if e:
        f = h5py.File('/data/civy/SMG/smg_data/CPDB_essential_test01_multiomics.h5', 'r')
    else:
        f = h5py.File('/data/civy/SMG/smg_data/CPDB_multiomics.h5', 'r')
    gene_name = f['gene_names'][...,-1].astype(str)
    label = np.logical_or(np.logical_or(f['y_test'][:], f['y_val'][:]), f['y_train'][:]).squeeze(-1)

    if h:      
        neg_label = gene_name[label]  
        label = pd.read_csv('/data/civy/SMG/smg_data/health.tsv', sep='\t').astype(str)['symbol'].tolist()
        mask = [gene_map[g] for g in sorted(list(set(label) & set(gene_list)))]
        np.random.seed(42)
        neg_mask = np.random.choice(sorted(list(set(np.arange(len(gene_list))) - set(mask))), min(len(mask),len(gene_list) - len(mask)) , replace=False).tolist()
    
    else:
        label = gene_name[label]  
        mask = [gene_map[g] for g in sorted(list(set(label) & set(gene_list)))]
        np.random.seed(42)
        neg_mask = np.random.choice(sorted(list(set(np.arange(len(gene_list))) - set(mask))), min(3 * len(mask),len(gene_list) - len(mask)) , replace=False).tolist()
    

    label = torch.zeros(len(gene_list), dtype=torch.float)

    label[mask] = 1
    labels = label.unsqueeze(1)

    final_mask = mask + neg_mask

    train_mask, test_mask, _, _ = train_test_split(final_mask, label[final_mask].numpy(), test_size = 0.2, shuffle=True, stratify=label[final_mask].numpy(), random_state=42)

    train_mask, val_mask, _, _ = train_test_split(train_mask, label[train_mask].numpy(), test_size = 0.2, shuffle=True, stratify=label[train_mask].numpy(),random_state=42)

    
    graph.ndata['feat'] = feats
    
    graph.ndata['label'] = labels
    
    
    index = torch.zeros(len(gene_list), dtype=torch.bool)
    index[train_mask] = 1

    
    graph.ndata['train_mask'] = index.unsqueeze(1)
    
    
    index = torch.zeros(len(gene_list), dtype=torch.bool)
    index[test_mask] = 1

    graph.ndata['test_mask'] = index.unsqueeze(1)
    
    index = torch.zeros(len(gene_list), dtype=torch.bool)
    index[val_mask] = 1

    graph.ndata['val_mask'] = index.unsqueeze(1)
    
    graph = dgl.add_self_loop(graph)
    
    return graph, (graph.ndata["feat"].shape[1], 1)
    
def get_new_ppi_I(e=False, h=False):

    high_conf_edgelist = pd.read_csv('/data/civy/SMG/smg_data/IREF_symbols_20190730.tsv', sep='\t')


    _, gene_list_1 = pd.factorize(high_conf_edgelist['partner1'])

    _, gene_list_2 = pd.factorize(high_conf_edgelist['partner2'])


    feature_list =  pd.read_csv('/data/civy/SMG/smg_data/multiomics_features.tsv', sep='\t')


    feature_name = feature_list['Unnamed: 0'].tolist() 
    

    gene_list = sorted(list((set(gene_list_1) | set(gene_list_2))))
    

    gene_map  = {g: i for i, g in enumerate(gene_list)}

    feature_map = {g: i for i, g in enumerate(feature_name)}

    src = [gene_map[g] for g in high_conf_edgelist['partner1'].tolist()]

    dst = [gene_map[g] for g in high_conf_edgelist['partner2'].tolist()]

    mask = [gene_map[g] for g in sorted(list(set(gene_list) & set(feature_name)))]

    feature = [feature_list.iloc[feature_map[g], 1:].tolist() for g in sorted(list(set(gene_list) & set(feature_name)))]


    feats =  torch.zeros((len(gene_map), 64))

    feats[mask] = torch.tensor(feature, dtype=torch.float)

    #print(feats)

    graph = dgl.graph((src, dst))

    #label = pd.read_csv('/home/yancui/ppimae/NCG_cancerdrivers_annotation_supporting_evidence.tsv', sep='\t')['symbol'].tolist()

    if e:
        f = h5py.File('/data/civy/SMG/smg_data/IRefIndex_2015_essential_test01_multiomics.h5', 'r')
    else:
        f = h5py.File('/data/civy/SMG/smg_data/IRefIndex_2015_multiomics.h5', 'r')

    gene_name = f['gene_names'][...,-1].astype(str)
    label = np.logical_or(np.logical_or(f['y_test'][:], f['y_val'][:]), f['y_train'][:]).squeeze(-1)


    if h:      
        neg_label = gene_name[label]  
        label = pd.read_csv('/data/civy/SMG/smg_data/health.tsv', sep='\t').astype(str)['symbol'].tolist()
        mask = [gene_map[g] for g in sorted(list(set(label) & set(gene_list)))]
        np.random.seed(42)
        neg_mask = np.random.choice(sorted(list(set(np.arange(len(gene_list))) - set(mask))), min(len(mask), len(gene_list) - len(mask)) , replace=False).tolist()
    
    else:
        label = gene_name[label]  
        mask = [gene_map[g] for g in sorted(list(set(label) & set(gene_list)))]
        np.random.seed(42)
        neg_mask = np.random.choice(sorted(list(set(np.arange(len(gene_list))) - set(mask))), min(3 * len(mask), len(gene_list) - len(mask)) , replace=False).tolist()
        
    #torch.use_deterministic_algorithms(True)
    
    label = torch.zeros(len(gene_list), dtype=torch.float)
    label[mask] = 1
    labels = label.unsqueeze(1)

    final_mask = mask + neg_mask

    
    train_mask, test_mask, _, _ = train_test_split(final_mask, label[final_mask].numpy(), test_size = 0.2, shuffle=True, stratify=label[final_mask].numpy(), random_state=42)
    
    train_mask, val_mask, _, _ = train_test_split(train_mask, label[train_mask].numpy(), test_size = 0.2, shuffle=True, stratify=label[train_mask].numpy(),random_state=42)

    graph.ndata['feat'] = feats

    graph.ndata['label'] = labels
    

    index = torch.zeros(len(gene_list), dtype=torch.bool)
    index[train_mask] = 1


    graph.ndata['train_mask'] = index.unsqueeze(1)

    index = torch.zeros(len(gene_list), dtype=torch.bool)
    index[test_mask] = 1

    graph.ndata['test_mask'] = index.unsqueeze(1)

    index = torch.zeros(len(gene_list), dtype=torch.bool)
    index[val_mask] = 1

    graph.ndata['val_mask'] = index.unsqueeze(1)
    
    graph = dgl.add_self_loop(graph)
    
    
    return graph, (graph.ndata["feat"].shape[1], 1)


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

    return model


from sklearn.model_selection import StratifiedKFold

from sklearn import metrics

#import zzz

def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
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

    set_random_seed(42)

    
    if args.ppi < 6:
        graph, (num_features, num_classes) = get_ppi(dataset=args.ppi, essential_gene=args.essential, health_gene=args.health) 
    elif args.ppi == 6:
        graph, (num_features, num_classes) = get_new_ppi_c(h=args.health, e=args.essential)
    elif args.ppi == 7:
        graph, (num_features, num_classes) = get_new_ppi_I(h=args.health, e=args.essential)
    

    #print(set(g.ndata['test_mask']) & set(graph.ndata['test_mask']))
    #graph, (num_features, num_classes), label_transfer, transfer_mask = get_ppi(dataset=args.ppi, essential_gene=args.essential, health_gene=args.health)       

    #graph, (num_features, num_classes) = get_new_ppi_I(h=True)
    #graph, (num_features, num_classes) = get_new_ppi_I(e=True)

    #raph, (num_features, num_classes) = get_ppi
    #graph, (num_features, num_classes) = get_ppi(dataset=args.ppi, essential_gene=args.essential, health_gene=args.health)

    train_mask = graph.ndata['train_mask'] 
    test_mask = graph.ndata['test_mask']
    val_mask = graph.ndata['val_mask']

    set_random_seed(42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    mask = (train_mask | test_mask | val_mask )
    
    print(val_mask.sum(), test_mask.sum(), train_mask.sum())
    

    '''g, (num_features, num_classes)  = get_health(dataset=Dataset[args.ppi])
    health_index =  torch.arange(graph.ndata['train_mask'].shape[0])[(g.ndata['test_mask'] |g.ndata['val_mask']).squeeze()]
    graph.ndata['train_mask'][health_index] = 0
    graph.ndata['val_mask'][health_index] = 0'''

    
    indices = np.arange(mask.shape[0])[mask.squeeze()]
     
    y = graph.ndata['label'].squeeze()[mask.squeeze()].numpy()
    
    
    args.num_features = num_features

    acc_list = []
    estp_acc_list = []
    
    if args.health:
        run_time = 10
    else:
        run_time = 10
    
    set_random_seed(42)
    for i, (train_ind, test_ind) in enumerate(skf.split(indices, y)):
  
        if i < run_time:

            if run_time == 10:
                train_index = indices[train_ind]
                test_index = indices[test_ind]
                
                train_index, val_index, _, _ = train_test_split(train_index, y[train_ind], test_size = 0.2, shuffle=True, stratify=y[train_ind], random_state=42) 
                
                index = torch.zeros(mask.shape[0], dtype=torch.bool)
                index[train_index] = 1

                graph.ndata['train_mask'] = index.unsqueeze(1)

                index = torch.zeros(mask.shape[0], dtype=torch.bool)
                index[test_index] = 1

                graph.ndata['test_mask'] = index.unsqueeze(1)

                index = torch.zeros(mask.shape[0], dtype=torch.bool)
                index[val_index] = 1

                graph.ndata['val_mask'] = index.unsqueeze(1)
            
            print(f"####### Run {0} for seed {0}")

            
            if logs:
                logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
            else:
                logger = None


            model = build_model(args).double()
            model.to(device)
            optimizer = create_optimizer(optim_type, model, lr, weight_decay)

            if use_scheduler and max_epoch !=0:
                logging.info("Use schedular")
                scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
            else:
                scheduler = None
            
                
            x = graph.ndata["feat"].double()
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
            #model.eval()


            scheduler_f = None   

            
            inducive_dataset = args.inductive_ppi
            final_acc, estp_acc, precision, recall, best_model = node_classification_evaluation(model.double(), inducive_dataset, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, scheduler_f, linear_prob=False)


            if logger is not None:
                logger.finish()

            output, embed = best_model(graph.to(device), x.to(device), return_hidden=True)
            
            
    
            #roc_auc, area, precision, recall = result(output[g.ndata['test_mask'].squeeze()].detach().cpu(), g.ndata['label'].squeeze().detach().cpu().numpy()[g.ndata['test_mask'].squeeze()])
            
            #print(area)
            
            acc_list.append(estp_acc)
            estp_acc_list.append(estp_acc)
    
    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    labels = graph.ndata["label"].to(device)
    
    
    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    #print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    #print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")
    if args.max_epoch > 2:
        args.encoder = "smg"
    #np.savetxt('{}_{}_fold1_precision.txt'.format(args.ppi, args.encoder), precision)
    #np.savetxt('{}_{}_fold1_recall.txt'.format(args.ppi, args.encoder), recall)
    print(f"# final_acc: {np.max(acc_list):.4f}/{np.median(acc_list):.4f}/{np.min(acc_list):.4f}")
    print(f"# final_acc: {np.mean(acc_list):.4f}/{np.std(acc_list):.4f}")
    
    return graph, x, best_model


import sys

import matplotlib.pyplot as plt
from dgl.nn.pytorch.explain import GNNExplainer
from captum.attr import IntegratedGradients
from dgl.nn import GraphConv
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from pandas import DataFrame
from collections import Counter

from collections import defaultdict

def construct_adjacency_list(edges):
    adjacency_list = defaultdict(list)
    for source, target in edges:
        adjacency_list[source].append(target)
        adjacency_list[target].append(source)  # Assuming an undirected graph
    return adjacency_list

def dfs(node, graph, visited, component):
    visited[node] = True
    component.append(node)
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs(neighbor, graph, visited, component)

def find_connected_components(graph):
    visited = {node: False for node in graph}
    components = []
    for node in graph:
        if not visited[node]:
            component = []
            dfs(node, graph, visited, component)
            components.append(component)
    return components

def find_max_connected_subgraph(edges):
    adjacency_list = construct_adjacency_list(edges)
    connected_components = find_connected_components(adjacency_list)
    max_component = max(connected_components, key=len)
    return max_component


import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.font_manager import FontProperties

def create_and_save_bar_plot(class_counter, save_path="pathway.png", title="Gene Node Frequency", x_label="Genes", y_label="Counts", rotation=45):
    # Set Arial font
    arial_font = FontProperties(fname="/home/yancui/.local/share/fonts/arial/arial.ttf", size=5)  # Replace with the actual path to Arial font file

    # Extract class labels and counts from the Counter
    class_labels = list(class_counter.keys())
    class_counts = list(class_counter.values())

    plt.figure(figsize=(10, 6))
    # Create the bar plot
    bars = plt.bar(class_labels, class_counts)

    for bar, label in zip(bars, class_labels):
        if label in ["GRB2","EGFR", "GAB1"]:
            bar.set_color('red')

    for bar, label in zip(bars, class_labels):
        if label in ["PIK3CA"]:
            bar.set_color('grey')
    # Set labels and title
    plt.xlabel(x_label, fontproperties=arial_font)
    plt.ylabel(y_label, fontproperties=arial_font)
    plt.title(title, fontproperties=arial_font)

    # Set font for tick labels
    plt.xticks(fontproperties=arial_font, rotation=rotation)
    plt.yticks(fontproperties=arial_font)

    # Save the plot
    plt.savefig(save_path, bbox_inches="tight", dpi=500)

def GNNexplain(graph, feat, model):
    explainer = GNNExplainer(model, num_hops=1)
    transform = dgl.ToSimple() 
    print(genes[graph.ndata['name'][20]])  
    new_center, sg, feat_mask, edge_mask = explainer.explain_node(20, graph, feat)
    dgl.save_graphs('sg.bin', [sg])
    torch.save(edge_mask, 'edge_mask.pt')
    sg.edata['mask'] = edge_mask
    sg.edata['rank'] = torch.argsort(-sg.edata['mask'])
    sg.ndata['degree'] = sg.in_degrees(sg.nodes())
    sg = transform(sg.to('cpu'))
    torch.save(feat_mask, 'feat_mask.pt')
    index = torch.argsort(sg.edata['rank']) < 50
    index = torch.arange(sg.edata['mask'].shape[0])[index]
    src, tgt = sg.edges()
    #print(genes[sg.ndata['name'][src[index]].squeeze(1)])
    c = Counter(genes[sg.ndata['name'][src[index]].squeeze(1)].tolist() + genes[sg.ndata['name'][tgt[index]].squeeze(1)].tolist())
    create_and_save_bar_plot(c)
    #edges = list(zip(genes[sg.ndata['name'][src[index]].squeeze(1)].tolist(),  genes[sg.ndata['name'][tgt[index]].squeeze(1)].tolist()))
    #print(edges)
    #max_connected_subgraph = find_max_connected_subgraph(edges)
    #print(max_connected_subgraph)
    '''G = dgl.to_networkx(sg.edge_subgraph(index), node_attrs=['name'], edge_attrs=['rank'])
    #seed = 13648  # Seed random number generators for reproducibility
    pos = nx.circular_layout(G)

    #node_sizes = sg.in_degrees(sg.nodes())
    #M = G.number_of_edges()
    #print(M)


    edge_colors = sg.edge_subgraph(index).edata['mask'].numpy()
    #edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    cmap = plt.cm.ocean
    #fig, ax = plt.subplots(figsize=(50, 20))
    node_size = [10 for node in list(G)]
    node_label = {i: genes[v] for i, v in list(G.nodes(data="name"))}
    #list(G.edges_iter())
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edge_cmap=cmap, alpha=0.4)
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        #with_labels=True,
        node_size = 50,
        node_color="gainsboro"
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels = node_label,
        font_size = 3,
        font_weight = 'bold'
    )
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = min(edge_colors), vmax=max(edge_colors)))
    plt.colorbar(sm)

    plt.savefig('Ex.png', format='png')
    plt.close()'''

    return
    
def extract_subgraph(g, node):
    seed_nodes = [node]
    sg = dgl.in_subgraph(g, seed_nodes)
    src, dst = sg.edges()
    seed_nodes = torch.cat([src, dst]).unique()
    sg = dgl.in_subgraph(g, seed_nodes, relabel_nodes=True)
    return sg


def IG(graph, feat, model):
    h = graph.ndata['feat'].clone().requires_grad_(True)
    ig = IntegratedGradients(partial(model.forward, graph))
    # Attribute the predictions for node class 0 to the input features
    feat_attr = ig.attribute(h, target=0, internal_batch_size=graph.num_nodes(), n_steps=50)
    node_weights = feat_attr.abs().sum(dim=1)
    torch.save(graph.ndata['name'], 'name.pt') 
    torch.save(node_weights, 'feat.pt')


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    set_random_seed(42)
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    try:
        graph, x, best_model = main(args)
        if args.GE:
            GNNexplain(graph.to('cuda'), x.to('cuda'), best_model.to('cuda'))
        if args.IGE:
            IG(graph.to('cuda'), x.to('cuda'), best_model.to('cuda'))
    except Exception:
        import traceback
        import pdb

        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
 