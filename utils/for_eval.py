from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, silhouette_score, adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np

import torch


def accuracy(model, ds, device, n_cluster, global_center):
    truth, feature = [], []
    model.eval()
    with torch.no_grad():
        for x, y in ds:
            x = x.to(device)
            x = x.float()
            feature.append(model.encoder(x).cpu())
            truth.append(y)
    if global_center is None or not global_center.any():
        kmeans = KMeans(n_cluster).fit(torch.cat(feature).numpy())
    else:
        kmeans = KMeans(n_cluster, init=global_center).fit(torch.cat(feature).numpy())
    y_pred = torch.tensor(kmeans.labels_)
    confusion_m = confusion_matrix(torch.cat(truth).numpy(), y_pred)
    _, col_idx = linear_sum_assignment(confusion_m, maximize=True)
    acc = np.trace(confusion_m[:, col_idx]) / confusion_m.sum()

    return acc

def accuracy1(model, ds, device):
    truth, pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in ds:
            x = x.to(device)
            x = x.float()
            truth.append(y)
            pred.append(model(x).max(1)[1].cpu())
    confusion_m = confusion_matrix(torch.cat(truth).numpy(), torch.cat(pred).numpy())
    _, col_idx = linear_sum_assignment(confusion_m, maximize=True)
    acc = np.trace(confusion_m[:, col_idx]) / confusion_m.sum()

    nmi_score = np.float64(normalized_mutual_info_score(torch.cat(truth).numpy(), torch.cat(pred).numpy()))

    # 计算调整兰德指数
    adjusted_rand_idx = np.float64(adjusted_rand_score(torch.cat(truth).numpy(), torch.cat(pred).numpy()))
    return acc, nmi_score, adjusted_rand_idx
