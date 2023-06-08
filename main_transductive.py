import logging
from webbrowser import get
from matplotlib import test
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
from openTSNE import TSNE

import utils
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


Dataset = ['CPDB', 'IRefIndex', 'PCNet', 'IRefIndex_2015', 'STRINGdb', 'Multinet']


f = h5py.File('/home/yancui/ppimae/{}_multiomics.h5'.format(Dataset[0]), 'r') 
genes = f['gene_names'][...,-1].astype(str)

np.save('genes', genes)

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

def get_health(dataset=0):
    k = pd.read_csv('/home/yancui/ppimae/graphmae/datasets/health.tsv', sep='\t').astype(str)
    f = h5py.File('/home/yancui/ppimae/{}_multiomics.h5'.format(dataset), 'r') 
    positive = k['symbol'].unique()
    name = [x.decode() for x in list(f['gene_names'][:][..., 1])]
    print(len(positive))
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
        f = h5py.File('/home/yancui/ppimae/{}_essential_test01_multiomics.h5'.format(dataset), 'r')
    else:
        f = h5py.File('/home/yancui/ppimae/{}_multiomics.h5'.format(dataset), 'r') 
    src, dst = np.nonzero(f['network'][:])
    graph = dgl.graph((src, dst))
    #datas = torch.load("/data/guest/MTGCN/data/str_fearures.pkl").to("cpu")
    #name = torch.from_numpy(f['gene_names'][:])
    graph.ndata['name'] = torch.arange(f['features'][:].shape[0]).unsqueeze(1)
    graph.ndata['feat'] = torch.from_numpy(f['features'][:])
    graph.ndata['train_mask'] = torch.from_numpy(f['mask_train'][:])
    graph.ndata['val_mask'] = torch.from_numpy(f['mask_test'][:])
    graph.ndata['test_mask'] = torch.from_numpy(f['mask_test'][:])
    full_mask = np.arange(graph.ndata['val_mask'].shape[0])[graph.ndata['val_mask'] | graph.ndata['train_mask']]
    graph.ndata['label'] = torch.from_numpy(np.logical_or(np.logical_or(f['y_test'][:], f['y_val'][:]), f['y_train'][:])).float()
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

    graph, (num_features, num_classes) = get_ppi(dataset=args.ppi, essential_gene=args.expression, health_gene=True)   
    args.num_features = num_features

    acc_list = []
    estp_acc_list = []
  

    #full_mask = np.arange(graph.ndata['val_mask'].shape[0])[graph.ndata['val_mask'] | graph.ndata['train_mask']]
    
    #kf = KFold(n_splits=5, random_state=None, shuffle=True)    

    for i in [0]:
        print(f"####### Run {i} for seed {0}")
        set_random_seed(i)
        
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
        final_acc, estp_acc, precision, recall, best_model = node_classification_evaluation(model, inducive_dataset, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, scheduler_f, linear_prob=False)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

        if logger is not None:
            logger.finish()

    output, embed = best_model(graph.to(device), x.to(device), return_hidden=True)
    
    
    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    labels = graph.ndata["label"].to(device)
    
    
    torch.save(embed[-1][train_mask], 'embed_{}_e.pt'.format(args.ppi))
    torch.save(labels[train_mask], 'y_{}_e.pt'.format(args.ppi))
    draw_tSNE(embed[-1][train_mask], labels[train_mask], args.ppi)
    
    
    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")
    if args.max_epoch > 2:
        args.encoder = "mae"
    #np.savetxt('{}_{}'.format(args.encoder, Dataset[args.ppi]), np.array(estp_acc_list))
    np.savetxt('{}_{}_pre'.format(args.encoder, Dataset[args.ppi]), precision)
    np.savetxt('{}_{}_recall'.format(args.encoder, Dataset[args.ppi]),  recall)
    return graph, x, best_model


from dgl.nn.pytorch.explain import GNNExplainer
from captum.attr import IntegratedGradients
from dgl.nn import GraphConv
from functools import partial



import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from pandas import DataFrame


def GNNexplain(graph, feat, model):
    explainer = GNNExplainer(model, num_hops=1)
    transform = dgl.ToSimple()   
    new_center, sg, feat_mask, edge_mask = explainer.explain_node(20, graph, feat)
    #print(sg.edges())
    #print(edge_mask)
    dgl.save_graphs('sg.bin', [sg])
    torch.save(edge_mask, 'edge_mask.pt')
    sg.edata['mask'] = edge_mask
    sg.edata['rank'] = torch.argsort(sg.edata['mask'])
    sg.ndata['degree'] = sg.in_degrees(sg.nodes())
    sg = transform(sg.to('cpu'))
    torch.save(feat_mask, 'feat_mask.pt')
    index = torch.argsort(sg.edata['rank']) < 80
    index = torch.arange(sg.edata['mask'].shape[0])[index]
    G = dgl.to_networkx(sg.edge_subgraph(index), node_attrs=['name'], edge_attrs=['rank'])
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
    #nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="indigo")
    # set alpha value for each edge

    #ax.margins(0.1, 0.05)
    #fig.tight_layout()
    #plt.axis("off")
    #plt.savefig('Ex.pdf', format='pdf')
    #plt.close()

    #feat_mask = feat_mask.to("cpu").detach()

    #feat = DataFrame({"Mutation": feat_mask[:16], "CNAs": feat_mask[16:32], "Methylation": feat_mask[32:48], "Expression":feat_mask[48:]})
    #print(feat.head())
    #cmapz = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
    #sns.heatmap(feat, cmap=cmapz)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = min(edge_colors), vmax=max(edge_colors)))
    plt.colorbar(sm)

    plt.savefig('Ex.png', format='png')
    plt.close()

    return
    
def extract_subgraph(g, node):
    seed_nodes = [node]
    sg = dgl.in_subgraph(g, seed_nodes)
    src, dst = sg.edges()
    seed_nodes = torch.cat([src, dst]).unique()
    sg = dgl.in_subgraph(g, seed_nodes, relabel_nodes=True)
    return sg


def IG(graph, feat, model):
    h = graph.ndata['feat'].clone().requires_grad_(True).float()
    ig = IntegratedGradients(partial(model.forward, graph))
    # Attribute the predictions for node class 0 to the input features
    feat_attr = ig.attribute(h, target=0, internal_batch_size=graph.num_nodes(), n_steps=50)
    node_weights = feat_attr.abs().sum(dim=1)
    torch.save(graph.ndata['name'], 'name.pt') 

import sys

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    try:
        graph, x, best_model = main(args)
        #GNNexplain(graph.to('cuda'), x.to('cuda'), best_model.to('cuda'))
        IG(graph.to('cuda'), x.to('cuda'), best_model.to('cuda'))
    except Exception:
        import traceback
        import pdb

        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
 