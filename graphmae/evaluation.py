import copy
from tqdm import tqdm
import torch
import torch.nn as nn

from graphmae.utils import create_optimizer, accuracy
from graphmae.models.gat import GAT
from graphmae.models.gin import GIN
from graphmae.models.gcn import GCN 
from graphmae.models.dot_gat import DotGAT
from graphmae.models.gcnii import GCNII
import h5py
import pandas as pd
import dgl

from sklearn import metrics

Dataset = ['CPDB', 'IRefIndex', 'PCNet', 'IRefIndex_2015', 'STRINGdb', 'Multinet']

def get_health(dataset):
    dataset = Dataset[dataset]
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
    return graph, (graph.ndata["feat"].shape[1], 2)



def result(pred, true):
    aa = torch.sigmoid(pred)
    precision, recall, _thresholds = metrics.precision_recall_curve(true, aa)
    area = metrics.auc(recall, precision)
    return metrics.roc_auc_score(true, aa), area, precision, recall

def node_classification_evaluation(model, inductive_dataset, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, sche, linear_prob=True, mute=False):
    model.eval()
    linear_prob = False
    if linear_prob:
        with torch.no_grad():
            x = model.embed(graph.to(device), x.to(device))
            torch.save(x, 'MAE.pt')
        encoder = LogisticRegression(in_feat, num_classes)
    else:
        encoder = model.encoder
        encoder.reset_classifier(num_classes)
        

    num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")
    
    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    final_aupr, estp_aupr, precision_f, recall_f, best_model = linear_probing_for_transductive_node_classiifcation(encoder, inductive_dataset, graph, x, optimizer_f, max_epoch_f, device, sche, mute)
    return final_aupr, estp_aupr, precision_f, recall_f, best_model


def node_classification_eval(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=True, mute=False):
    model.eval()
    linear_prob = False
    if linear_prob:
        with torch.no_grad():
            x = model.embed(graph.to(device), x.to(device))
            torch.save(x, 'mae.pt')
        encoder = LogisticRegression(in_feat, num_classes)
    else:
        encoder = model.encoder
        encoder.reset_classifier(num_classes, concat=True, datas_dim=256)
        
    num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")
    
    

    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    final_auc, estp_auc = Concat(encoder, graph, x, optimizer_f, max_epoch_f, device, mute)
    return final_auc, estp_auc


def node_classification(model, etype, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=True, mute=False):
    #model.eval()
    #with torch.no_grad():
        #x = model.embed(graph.to(device), x.to(device))

    #etype = 'gat'
    if(etype == 'gat'):
        gat = GAT(in_dim=64,
                    num_hidden=64,
                    out_dim=64,
                    num_layers=3,
                    nhead=4,
                    nhead_out=4,
                    activation='prelu',
                    feat_drop=.2,
                    attn_drop=.1,
                    negative_slope=0.2,
                    residual=False,
                    norm=None,
                    concat_out=True)
        gat.reset_classifier(num_classes)
    if(etype == 'gin'):
        gat = GIN(in_dim=64,
                 num_hidden=256,
                 out_dim=256,
                 num_layers=3,
                 dropout= 0.1,
                 activation='relu',
                 residual=True,
                 norm="layernorm",
                 encoding=False,
                 learn_eps=False,
                 aggr="mean",
                 )
        gat.reset_classifier(num_classes)
    if(etype == 'gcn'):
        gat = GCN(
                 in_dim=64,
                 num_hidden=256,
                 out_dim=256,
                 num_layers=3,
                 dropout=0.3,
                 activation='relu',
                 residual=True,
                 norm=None,
                 encoding=False
                 )
        gat.reset_classifier(num_classes) 
    if(etype == 'dotgat'):
        gat = GIN(in_dim=64,
                    num_hidden=64,
                    out_dim=64,
                    num_layers=3,
                    nhead=4,
                    nhead_out=4,
                    activation='prelu',
                    feat_drop=.2,
                    attn_drop=.1,
                    negative_slope=0.2,
                    residual=False,
                    norm=None,
                    concat_out=True)
        gat.reset_classifier(num_classes) 

    num_finetune_params = [p.numel() for p in gat.parameters() if  p.requires_grad]
    if not mute:
        print(f"num parameters for fsinetuning: {sum(num_finetune_params)}")
    
    gat.to(device)
    optimizer_f = create_optimizer("adam", gat, lr_f, weight_decay_f)
    final_auc, estp_auc = linear_probing_for_transductive_node_classiifcation(gat, graph, x, optimizer_f, max_epoch_f, device, mute)
    return final_auc, estp_auc


import torch
import torch.nn.functional as F


def sigmoid_focal_loss(
    inputs,
    targets,
    alpha: float = 0.01,
    gamma: float = 2,
    reduction: str = "sum",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.max(1)[0].float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss



import numpy as np

def linear_probing_for_transductive_node_classiifcation(model, inductive_dataset, graph, feat, optimizer, max_epoch, device, sche, mute=False):
    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    labels = graph.ndata["label"].to(device)
    
    pos_num = labels[train_mask].sum()
    neg_num = train_mask.sum() - pos_num

    print(pos_num, neg_num)
    
    weight = torch.tensor([neg_num/pos_num])
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight).to(device)
    if inductive_dataset !=-1:
        inductive = True
    else:
        inductive =False
    graph = graph.to(device)
    x = feat.to(device)


    best_val_aupr = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)
        
    #g, _ = get_ppi(dataset=inductive_dataset)
    #g, _ = get_ppi(dataset=inductive_dataset)
    #feat = g.ndata['feat'].float()
    #m = g.ndata['val_mask']
    #l = g.ndata['label'].to(device)


    for epoch in epoch_iter:
        model.train().to(device)
        out = model(graph, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()
        if sche is not None:
            sche.step()
        
        with torch.no_grad():
            model.eval()
            if inductive:
                #print(pred[val_mask == 1].shape, labels[val_mask == 1].shape)
                pred = model(g.to(device), feat.to(device))
                val_auc, val_aupr = result(pred[m].cpu(), l[m].cpu())
                val_loss = criterion(pred[m], l[m])
                test_auc, test_aupr = result(pred[m].cpu(), l[m].cpu())
                test_loss = criterion(pred[m], l[m])
            else:
                pred = model(graph, x)
                #print(pred[val_mask == 1].shape, labels[val_mask == 1].shape)
                val_auc, val_aupr, precision, recall = result(pred[val_mask].cpu(), labels[val_mask].cpu())
                val_loss = criterion(pred[val_mask], labels[val_mask])
                test_auc, test_aupr, precision, recall = result(pred[test_mask].cpu(), labels[test_mask].cpu())
                test_loss = criterion(pred[test_mask], labels[test_mask])         
        
        if val_aupr >= best_val_aupr:
            best_val_aupr = val_aupr
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_auc:{val_aupr}, test_loss:{test_loss.item(): .4f}, test_auc:{test_aupr: .4f}")

    best_model.eval()
    with torch.no_grad():
        if inductive:
            g, _ = get_health(dataset=0)
            feat = g.ndata['feat'].float()
            pred = model(g.to(device), feat.to(device))
            m = g.ndata['val_mask']
            l = g.ndata['label']
            estp_test_auc, estp_test_aupr = result(pred[m].cpu(), l[m].cpu())
        else:
            pred = best_model(graph, x)
            estp_test_auc, estp_test_aupr, precision_f, recall_f = result(pred[test_mask==1].cpu(), labels[test_mask==1].cpu())
    if mute:
        print(f"# IGNORE: --- Testauc: {test_aupr:.4f}, early-stopping-Testauc: {estp_test_aupr:.4f}, Best Valauc: {best_val_aupr:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(f"--- Testaupr: {test_aupr:.4f}, early-stopping-Testaupr: {estp_test_aupr:.4f}, Best Valaupr: {best_val_aupr:.4f} in epoch {best_val_epoch} --- ")

    # (final_auc, es_auc, best_auc)
    return estp_test_aupr, estp_test_aupr, precision_f, recall_f, best_model

def Concat(model, graph, feat, optimizer, max_epoch, device, mute=False):
    criterion = nn.BCEWithLogitsLoss()

    graph = graph.to(device)
    x = feat.to(device)

    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    labels = graph.ndata["label"]

    best_val_auc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)
    
    
    datas = torch.load("/data/guest/MTGCN/dgl.pkl_256").to("cpu")
    for epoch in epoch_iter:
        model.train()
        out = model(graph, x, datas=datas, concat=True)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(graph, x, datas=datas, concat=True)
            #print(pred[val_mask == 1].shape, labels[val_mask == 1].shape)
            val_auc, val_aupr = result(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_auc, test_aupr = result(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])
        
        if val_auc >= best_val_auc:
            best_val_auc = val_auc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_auc:{val_auc}, test_loss:{test_loss.item(): .4f}, test_auc:{test_auc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(graph, x, datas=datas, concat=True)
        estp_test_auc, estp_test_aupr = result(pred[test_mask==1], labels[test_mask==1])
    if mute:
        print(f"# IGNORE: --- Testauc: {test_auc:.4f}, early-stopping-Testauc: {estp_test_auc:.4f}, Best Valauc: {best_val_auc:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(f"--- Testauc: {test_auc:.4f}, early-stopping-Testauc: {estp_test_auc:.4f}, Best Valauc: {best_val_auc:.4f} in epoch {best_val_epoch} --- ")

    # (final_auc, es_auc, best_auc)
    return test_auc, estp_test_auc


def linear_probing_for_inductive_node_classiifcation(model, x, labels, mask, optimizer, max_epoch, device, mute=False):
    if len(labels.shape) > 1:
        criterion = torch.nn.BCEWithLogitsLoss(label_s)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    train_mask, val_mask, test_mask = mask

    best_val_auc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)  

        best_val_auc = 0

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(None, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(None, x)
            val_auc, _ = result(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_auc, _ = result(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])
        
        if val_auc >= best_val_auc:
            best_val_auc = val_auc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_auc:{val_auc}, test_loss:{test_loss.item(): .4f}, test_auc:{test_auc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(None, x)
        estp_test_auc = aucuracy(pred[test_mask], labels[test_mask])
    if mute:
        print(f"# IGNORE: --- Testauc: {test_auc:.4f}, early-stopping-Testauc: {estp_test_auc:.4f}, Best Valauc: {best_val_auc:.4f} in epoch {best_val_epoch} ")
    else:
        print(f"--- Testauc: {test_auc:.4f}, early-stopping-Testauc: {estp_test_auc:.4f}, Best Valauc: {best_val_auc:.4f} in epoch {best_val_epoch}")

    return test_auc, estp_test_auc


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits
