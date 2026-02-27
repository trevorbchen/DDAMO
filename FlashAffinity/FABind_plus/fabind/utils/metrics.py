import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _binary_metrics(y_pred, y_true, threshold=0.5):
    y_true_np = _to_numpy(y_true).astype(bool)
    y_pred_np = _to_numpy(y_pred)
    y_pred_bin = (y_pred_np >= threshold)

    acc = accuracy_score(y_true_np, y_pred_bin)
    try:
        auroc = roc_auc_score(y_true_np.astype(int), y_pred_np)
    except Exception:
        auroc = float("nan")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_np.astype(int),
        y_pred_bin.astype(int),
        labels=[0, 1],
        average=None,
        zero_division=0,
    )
    return acc, auroc, precision, recall, f1

def myMetric(y_pred, y, threshold=0.5):
    y = y.float()
    criterion = nn.BCELoss()
    with torch.no_grad():
        loss = criterion(y_pred, y)

    y = y.bool()
    acc, auroc, precision, recall, f1 = _binary_metrics(y_pred, y, threshold=threshold)
    precision_0, precision_1 = precision
    recall_0, recall_1 = recall
    f1_0, f1_1 = f1
    return {"BCEloss":loss.item(),
            "acc":acc, "auroc":auroc, "precision_1":precision_1,
           "recall_1":recall_1, "f1_1":f1_1,"precision_0":precision_0,
           "recall_0":recall_0, "f1_0":f1_0}

def cls_metric(y_pred, y, threshold=0.5):
    y = y.float()
    criterion = nn.BCELoss()
    with torch.no_grad():
        loss = criterion(y_pred, y)

    y = y.bool()
    acc, auroc, precision, recall, f1 = _binary_metrics(y_pred, y, threshold=threshold)
    precision_0, precision_1 = precision
    recall_0, recall_1 = recall
    f1_0, f1_1 = f1
    return {"pocket_BCEloss":loss.item(),
            "pocket_acc":acc, "pocket_auroc":auroc, "pocket_precision_1":precision_1,
           "pocket_recall_1":recall_1, "pocket_f1_1":f1_1,"pocket_precision_0":precision_0,
           "pocket_recall_0":recall_0, "pocket_f1_0":f1_0}

def affinity_metrics(affinity_pred, affinity):
    affinity_pred_np = _to_numpy(affinity_pred)
    affinity_np = _to_numpy(affinity)
    try:
        pearson = np.corrcoef(affinity_pred_np, affinity_np)[0, 1]
    except Exception:
        pearson = float("nan")
    rmse = mean_squared_error(affinity_np, affinity_pred_np, squared=False)
    return {"pearson":pearson, "rmse":rmse}

def pocket_metrics(pocket_coord_pred, pocket_coord):
    pred_np = _to_numpy(pocket_coord_pred)
    coord_np = _to_numpy(pocket_coord)
    pearson_x = np.corrcoef(pred_np[:, 0], coord_np[:, 0])[0, 1]
    rmse_x = mean_squared_error(coord_np[:, 0], pred_np[:, 0], squared=False)
    mae_x = mean_absolute_error(coord_np[:, 0], pred_np[:, 0])
    pearson_y = np.corrcoef(pred_np[:, 1], coord_np[:, 1])[0, 1]
    rmse_y = mean_squared_error(coord_np[:, 1], pred_np[:, 1], squared=False)
    mae_y = mean_absolute_error(coord_np[:, 1], pred_np[:, 1])
    pearson_z = np.corrcoef(pred_np[:, 2], coord_np[:, 2])[0, 1]
    rmse_z = mean_squared_error(coord_np[:, 2], pred_np[:, 2], squared=False)
    mae_z = mean_absolute_error(coord_np[:, 2], pred_np[:, 2])
    pearson = (pearson_x + pearson_y + pearson_z) / 3
    rmse = (rmse_x + rmse_y + rmse_z) / 3
    mae = (mae_x + mae_y + mae_z) / 3
    pocket_pairwise_dist = F.pairwise_distance(pocket_coord_pred, pocket_coord, p=2)
    DCC = (pocket_pairwise_dist < 4).sum().item() / len(pocket_pairwise_dist)
    return {"pocket_pearson":pearson, "pocket_rmse":rmse, "pocket_mae":mae, "pocket_center_avg_dist": pocket_pairwise_dist.mean().item(), "pocket_center_DCC": DCC * 100}
    
def pocket_direct_metrics(pocket_coord_pred, pocket_coord):
    pred_np = _to_numpy(pocket_coord_pred)
    coord_np = _to_numpy(pocket_coord)
    pearson_x = np.corrcoef(pred_np[:, 0], coord_np[:, 0])[0, 1]
    rmse_x = mean_squared_error(coord_np[:, 0], pred_np[:, 0], squared=False)
    mae_x = mean_absolute_error(coord_np[:, 0], pred_np[:, 0])
    pearson_y = np.corrcoef(pred_np[:, 1], coord_np[:, 1])[0, 1]
    rmse_y = mean_squared_error(coord_np[:, 1], pred_np[:, 1], squared=False)
    mae_y = mean_absolute_error(coord_np[:, 1], pred_np[:, 1])
    pearson_z = np.corrcoef(pred_np[:, 2], coord_np[:, 2])[0, 1]
    rmse_z = mean_squared_error(coord_np[:, 2], pred_np[:, 2], squared=False)
    mae_z = mean_absolute_error(coord_np[:, 2], pred_np[:, 2])
    pearson = (pearson_x + pearson_y + pearson_z) / 3
    rmse = (rmse_x + rmse_y + rmse_z) / 3
    mae = (mae_x + mae_y + mae_z) / 3
    return {"pocket_direct_pearson":pearson, "pocket_direct_rmse":rmse, "pocket_direct_mae":mae}

def print_metrics(metrics):
    out_list = []
    for key in metrics:
        try:
            out_list.append(f"{key}:{metrics[key]:6.3f}")
        except:
            out_list.append(f"\n{key}:\n{metrics[key]}")
    out = ", ".join(out_list)
    return out


def compute_individual_metrics(pdb_list, inputFile_list, y_list):
    r_ = []
    for i in range(len(pdb_list)):
        pdb = pdb_list[i]
        # inputFile = f"{pre}/input/{pdb}.pt"
        inputFile = inputFile_list[i]
        y = y_list[i]
        (coords, y_pred, protein_nodes_xyz, 
         compound_pair_dis_constraint, pdb, sdf_fileName, mol2_fileName, pre) = torch.load(inputFile)
        result = myMetric(torch.tensor(y_pred).reshape(-1), y.reshape(-1))
        for key in result:
            result[key] = float(result[key])
        result['idx'] = i
        result['pdb'] = pdb
        result['p_length'] = protein_nodes_xyz.shape[0]
        result['c_length'] = coords.shape[0]
        result['y_length'] = y.reshape(-1).shape[0]
        result['num_contact'] = int(y.sum())
        r_.append(result)
    result = pd.DataFrame(r_)
    return result

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report