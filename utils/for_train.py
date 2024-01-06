import copy

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn
from .nn import get_p
from .for_eval import accuracy, accuracy1

DATA_PLOT = None


def set_data_plot(tr_ds, test_ds, device):
    global DATA_PLOT

    # select 100 sample per class  
    tr_x, tr_y = [], []
    count = torch.zeros(10, dtype=torch.int)
    for batch in tr_ds:
        for data, label in zip(*batch):
            tr_x.append(data[None])
            tr_y.append(label[None])
            count[label] += 1
    tr_x, tr_y = torch.cat(tr_x).to(device), torch.cat(tr_y).to(device)

    # select 100 sample per class
    test_x, test_y = [], []
    count = torch.zeros(10, dtype=torch.int)
    for batch in test_ds:
        for data, label in zip(*batch):
            test_x.append(data[None])
            test_y.append(label[None])
            count[label] += 1
    test_x, test_y = torch.cat(test_x).to(device), torch.cat(test_y).to(device)

    DATA_PLOT = {'train': (tr_x, tr_y),
                 'test': (test_x, test_y)}



def get_initial_center(model, ds, device, n_cluster, save_dir, c, global_center):
    # fit
    print('\nbegin fit kmeans++ to get initial cluster centroids ...')
    acc_h = History('max')
    truth, feature = [], []
    model.eval()
    with torch.no_grad():
        feature = []
        for x, y in ds:
            x = x.to(device)
            x = x.float()
            truth.append(y)
            feature.append(model.encoder(x).cpu())
    # feature_2D = torch.cat(feature).numpy()
    # feature_2D = TSNE(2).fit_transform(torch.cat(feature).numpy())
    # arr_truth = torch.cat(truth).numpy()
    # y_pred, y_border, center_num ,dc_percent, dc, local_center= DenPeakCluster(arr_truth, feature_2D, 10)
    # np.savetxt("features.txt", feature_2D)
    # print("features shape:",feature_2D.shape)
    # print('dc_percent: ', dc_percent)
    # print('dc: ', dc)
    # print('center: ', local_center)
    # print('center: ', type(local_center))
    # y_pred, centers = fit(feature_2D, 0.02, 4)
    # print('center: ', centers)
    # print('clusterIndex: ', y_pred)


    if global_center is None or not global_center.any():
        kmeans = KMeans(n_cluster).fit(torch.cat(feature).numpy())
    else:
        kmeans = KMeans(n_cluster, init=global_center).fit(torch.cat(feature).numpy())
    y_pred = kmeans.labels_
    local_center = kmeans.cluster_centers_
    # print('center: ', local_center)
    # print('center: ', type(local_center))

    # feature_2D = TSNE(2).fit_transform(torch.cat(feature).numpy())  # 可视化降维，降为2维
    # plt.scatter(feature_2D[:, 0], feature_2D[:, 1], 16, arr_truth, cmap='Paired')  # 根据预测标签染色画图
    # plt.title(f'Epoch1')
    # plt.savefig(f'{save_dir}/epoch_{c}_start.png')
    # plt.close()
    #
    # feature_2D = TSNE(2).fit_transform(torch.cat(feature).numpy())  # 可视化降维，降为2维
    # plt.scatter(feature_2D[:, 0], feature_2D[:, 1], 16, y_pred, cmap='Paired')  # 根据预测标签染色画图
    # plt.title(f'Epoch')
    # plt.savefig(f'{save_dir}/epoch_{c}.png')
    # plt.close()

    # confusion_m = confusion_matrix(arr_truth, y_pred)
    # _, col_idx = linear_sum_assignment(confusion_m, maximize=True)
    # acc = np.trace(confusion_m[:, col_idx]) / confusion_m.sum()
    # # acc_h.add(acc)
    # print(f'acc : {acc:.4f}  max acc : {acc_h.best:.4f}')
    # plt.cla()
    # plt.scatter(feature_2D[:, 0], feature_2D[:, 1], c=y_pred, s=0.5, alpha=0.5)
    # plt.scatter(local_center[:, 0], local_center[:, 1], c='red', marker='X', s=10, label='Centers')  # 使用红色的 X 标记表示聚类中心
    # plt.savefig(save_dir+f'/merged cluster_{c}.png')
    # np.savetxt(save_dir+f'/dc_coeff_{c}.txt', [dc_percent, dc, center_num]) 
    # macc = np.round(acc(arr_truth, y_pred), 5)
    # mnmi = np.round(nmi(arr_truth, y_pred), 5)
    # mari = np.round(ari(arr_truth, y_pred), 5)
    # print('Iter %d: acc=%.5f, nmi=%.5f, ari=%.5f;' % (0, macc, mnmi, mari))
    # #选择一个文件名来保存数据
    # file_name = f'output_{c}.txt'
    # with open(file_name, "w") as file:
    #     # 将center写入文件
    #     file.write("center: ")
    #     file.write(", ".join(map(str, local_center)))
    #     file.write("\n")
    #     file.write("truth: ")
    #     file.write(", ".join(map(str, arr_truth)))
    #     file.write("\n")
    #     file.write("y_pre: ")
    #     file.write(", ".join(map(str, y_pred)))
    #     file.write("\n")
    #     file.write("y_border: ")
    #     file.write(", ".join(map(str, y_border)))
    #     file.write("\n")
    #     file.write("acc: ")
    #     file.write(", ".join(macc.astype(str)))
    #     file.write("\n")
    #     file.write("nmi: ")
    #     file.write(", ".join(mnmi.astype(str)))
    #     file.write("\n")
    #     file.write("ari: ")
    #     file.write(", ".join(mari.astype(str)))
    #     file.write("\n")


    return local_center

def pretrain(model, opt, ds, device, epochs, save_dir, client):
    print('begin train AutoEncoder ...')
    
    loss_fn = nn.MSELoss()
    n_sample, n_batch = len(ds.dataset), len(ds)
    model.train() 
    loss_h = History('min')
    
    # fine-tune
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}:')
        print('-' * 10)
        loss = 0.
        for i, (x, y) in enumerate(ds, 1):
            opt.zero_grad()
            x = x.to(device)
            # print('x: ', x.shape) # x:  torch.Size([256, 10000])
            x = x.float()
            _, gen = model(x)
            # print('gen: ', gen.shape)            
            batch_loss = loss_fn(x, gen)
            batch_loss.backward()
            opt.step()
            loss += batch_loss * y.numel()
            print(f'{i}/{n_batch}', end='\r')

        loss /= n_sample
        loss_h.add(loss)
        print(f'loss : {loss.item():.4f}  min loss : {loss_h.best.item():.4f}')
        print(f'lr: {opt.param_groups[0]["lr"]}')
    plt.clf()
    plt.figure()
    plt.title('Pre_Training Loss vs Communication rounds')
    plt.plot(range(epochs), loss_h.history, color='r')
    plt.ylabel('Pre_Training loss')
    plt.xlabel('Epoch')
    plt.savefig(f'{save_dir}/loss_pretrain_epoch_{epoch}.png')
    plt.close()
    return model.state_dict()
                   

def train(model, opt, ds, device, epochs, save_dir, client):
    print('begin train DEC ...')
    loss_fn = nn.KLDivLoss(reduction='batchmean')
    n_sample, n_batch = len(ds.dataset), len(ds)
    loss_h, acc_h, nmi_h, adjusted_rand_idx_h = History('min'), History('max'), History('max'), History('max')
    
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}:')
        print('-' * 10)
        model.train()
        loss = 0.
        for i, (x, y) in enumerate(ds, 1):
            opt.zero_grad()
            x = x.to(device)        
            x = x.float()    
            q = model(x)
            batch_loss = loss_fn(q.log(), get_p(q))
            batch_loss.backward()
            opt.step()
            loss += batch_loss * y.numel()
            print(f'{i}/{n_batch}', end='\r')
            
        loss /= n_sample
        loss_h.add(loss)
        if loss_h.better:
            torch.save(model, f'{save_dir}/DEC_{client}.pt')
        print(f'loss : {loss.item():.4f}  min loss : {loss_h.best.item():.4f}')

        acc, nmi, adjusted_rand_idx = accuracy1(model, ds, device)
        acc_h.add(acc)
        nmi_h.add(nmi)
        adjusted_rand_idx_h.add(adjusted_rand_idx)
        print(f'acc : {acc:.4f}  max acc : {acc_h.best:.4f}')
        print(f'nmi : {nmi:.4f}  max nmi : {nmi_h.best:.4f}')
        print(f'rand_idx : {adjusted_rand_idx:.4f}  max rand_idx : {adjusted_rand_idx_h.best:.4f}')
        print(f'lr: {opt.param_groups[0]["lr"]}')
        
        # if epoch % 5 == 0:
        #     plot(model, save_dir, tr_ds, 'train', epoch)
                   
    df = pd.DataFrame(zip(range(1, epoch+1), loss_h.history, acc_h.history, nmi_h.history), columns=['epoch', 'loss', 'acc', 'nmi'])
    df.to_excel(f'{save_dir}/train.xlsx', index=False)
    plt.clf()
    plt.figure()
    plt.title(f'DEC Training Loss epochs:{epochs}')
    plt.plot(range(epochs), loss_h.history, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Epoch')
    plt.savefig(f'{save_dir}/loss_train_epoch_{epoch}.png')
    plt.close()

    plt.clf()
    plt.figure()
    plt.title(f'DEC Training Rand epochs:{epochs}')
    plt.plot(range(epochs), adjusted_rand_idx_h.history, color='r')
    plt.ylabel('Training Rand')
    plt.xlabel('Epoch')
    plt.savefig(f'{save_dir}/adjusted_rand_idx_train_epoch_{epoch}_clients_{client}.png')
    plt.close()

    plt.clf()
    plt.figure()
    plt.title(f'DEC Training NMI epochs:{epochs}')
    plt.plot(range(epochs), nmi_h.history, color='r')
    plt.ylabel('Training NMI')
    plt.xlabel('Epoch')
    plt.savefig(f'{save_dir}/NMI_train_epoch_{epoch}_clients_{client}.png')
    plt.close()

    plt.clf()
    plt.figure()
    plt.title(f'DEC Training Acc epochs:{epochs}')
    plt.plot(range(epochs), acc_h.history, color='r')
    plt.ylabel('Training acc')
    plt.xlabel('Epoch')
    plt.savefig(f'{save_dir}/acc_train_epoch_{epoch}_clients_{client}.png')
    plt.close()
    return acc_h.best,nmi_h.best,adjusted_rand_idx_h.best
               
class History:
    def __init__(self, target='min'):
        self.value = None
        self.best = float('inf') if target == 'min' else 0.
        self.n_no_better = 0
        self.better = False
        self.target = target
        self.history = [] 
        self._check(target)
        
    def add(self, value):
        if self.target == 'min' and value < self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        elif self.target == 'max' and value > self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        else:
            self.n_no_better += 1
            self.better = False
            
        self.value = value
        self.history.append(value.item())
        
    def _check(self, target):
        if target not in {'min', 'max'}:
            raise ValueError('target only allow "max" or "min" !')
    

def plot(model, save_dir, target='train', epoch=None):
    from matplotlib import pyplot as plt1
    assert target in {'train', 'test'}
    print('plotting ...')
    
    model.eval()
    with torch.no_grad():
        feature = model.encoder(DATA_PLOT[target][0])  # DATA_PLOT[target][0]为训练数据集输入，feature获取编码特征
        pred = model.cluster(feature).max(1)[1].cpu().numpy()  # 预测聚类结果，返回预测矩阵
    feature_2D = TSNE(2).fit_transform(feature.cpu().numpy())  # 可视化降维，降为2维
    plt1.clf()
    # plt1.rcParams['figure.figsize']=(6.4, 4.8)
    plt.figure(figsize=(6.4,4.8))
    plt1.scatter(feature_2D[:, 0], feature_2D[:, 1], 16, pred, cmap='Paired')  # 根据预测标签染色画图
    # plt1.xlim(min(feature_2D[:, 0]), max(feature_2D[:, 0]))
    # plt1.ylim(min(feature_2D[:, 1]), max(feature_2D[:, 1]))
    # # print('pred: ', min(feature_2D[:, 0]),max(feature_2D[:, 0]))
    # print('x: ', min(feature_2D[:, 0]),max(feature_2D[:, 0]))
    # print('y: ', min(feature_2D[:, 1]), max(feature_2D[:, 1]))
    if epoch is None:
        plt1.title(f'Test data',fontsize=15)
        plt1.savefig(f'{save_dir}/test.png')
    else:
        plt1.title(f'Epoch: {epoch}',fontsize=15)
        plt1.savefig(f'{save_dir}/epoch_{epoch}.png')
    plt1.close()


def average_global_weights(w):  # 返回权重的平均值，即执行联邦平均算法：
    """
    这个w是经过多轮本地训练后统计的权重list，在参数默认的情况下，是一个长度为10的列表，而每个元素都是一个字典，
    每个字典都包含了模型参数的名称（比如layer_input.weight或者layer_hidden.bias），以及其权重具体的值。
    """
    w_avg = copy.deepcopy(w[0])
    # 深复制函数深复制，就是从输入变量完全复刻一个相同的变量，无论怎么改变新变量，原有变量的值都不会受到影响。这里复制了第一个用户的权重字典。
    # 随后，对于每一类参数进行循环，累加每个用户模型里对应参数的值，最后取平均获得平均后的模型。
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


from scipy.spatial.distance import cdist

def find_similar_centers(centers1, centers2, threshold):
    distances = cdist(centers1, centers2)
    grouped_centers = []
    for i in range(len(centers1)):
        group = [i]
        for j in range(len(centers1), len(centers1) + len(centers2)):
            if distances[i, j - len(centers1)] < threshold:
                group.append(j)
        grouped_centers.append(group)
    return grouped_centers



def merge_similar_centers(centers, grouped_centers):
    merged_centers = []
    for group in grouped_centers:
        group_centers = centers[group]
        if len(group_centers) == 1:
            merged_centers.append(group_centers[0])
        else:
            merged_center = np.mean(group_centers, axis=0)
            merged_centers.append(list(merged_center))
    return merged_centers



def merge_kmeans_centers(centers1, centers2, threshold):
    # 标记聚类中心所属组别
    marked_centers1 = np.c_[centers1, np.zeros(len(centers1))]
    marked_centers2 = np.c_[centers2, np.ones(len(centers2))]
    # 合并聚类中心
    centers = np.concatenate((marked_centers1, marked_centers2))
    # 计算相似聚类中心并合并
    grouped_centers = find_similar_centers(centers1, centers2, threshold)
    merged_centers = merge_similar_centers(centers[:, :-1], grouped_centers)
    return merged_centers

import torch


def fedprox(local_w, lr=0.01, mu=0.1, lambda_=0.01):
    # 客户端模型数量
    n_clients = len(local_w)
    # 初始化全局模型
    global_w = {}
    for k in local_w[0].keys():
        global_w[k] = torch.zeros_like(local_w[0][k])
    # 模型聚合
    for k in global_w.keys():
        for i in range(n_clients):
            if i == 0:
                global_w[k] += local_w[i][k]
            else:
                global_w[k] += (1 / n_clients) * (local_w[i][k] - (mu / lambda_) * (local_w[i][k] - local_w[i-1][k]))
        # 加上罚项
        global_w[k] -= lambda_ * lr * global_w[k]
    return global_w


import torch


def fedavg_plus(local_w, mu=0.9):
    # 客户端模型数量
    n_clients = len(local_w)
    # 初始化全局模型
    global_w = {}
    for k in local_w[0].keys():
        global_w[k] = torch.zeros_like(local_w[0][k])
    # 动态加权平均聚合
    for k in global_w.keys():
        # 初始化动态权重参数
        theta = torch.zeros(n_clients)
        for i in range(n_clients):
            theta[i] = torch.norm(local_w[i][k] - global_w[k])
        # 计算权重
        alpha = 1 / (1 + torch.exp(-mu * theta))
        alpha = alpha / torch.sum(alpha)
        # 聚合权重计算
        for i in range(n_clients):
            for key in global_w.keys():
                if i == 0:
                    global_w[key] += alpha[i] * local_w[i][key]
                else:
                    global_w[key] += alpha[i] * (local_w[i][key] - local_w[i-1][key]) + global_w[key]
    # 更新全局模型
    for key in global_w.keys():
        global_w[key] = global_w[key] / n_clients
    return global_w


from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_centers(centers1, centers2, threshold):
    distances = cdist(centers1, centers2)
    grouped_centers = []
    for i in range(len(centers1)):
        group = [i]
        for j in range(len(centers1), len(centers1) + len(centers2)):
            if distances[i, j - len(centers1)] < threshold:
                group.append(j)
        grouped_centers.append(group)
    return grouped_centers



def merge_similar_centers(centers, grouped_centers):
    merged_centers = []
    for group in grouped_centers:
        group_centers = centers[group]
        if len(group_centers) == 1:
            merged_centers.append(group_centers[0])
        else:
            merged_center = np.mean(group_centers, axis=0)
            merged_centers.append(list(merged_center))
    return merged_centers



def merge_kmeans_centers(centers1, centers2, threshold):
    # 标记聚类中心所属组别
    marked_centers1 = np.c_[centers1, np.zeros(len(centers1))]
    marked_centers2 = np.c_[centers2, np.ones(len(centers2))]
    # 合并聚类中心
    centers = np.concatenate((marked_centers1, marked_centers2))
    # 计算相似聚类中心并合并
    grouped_centers = find_similar_centers(centers1, centers2, threshold)
    merged_centers = merge_similar_centers(centers[:, :-1], grouped_centers)
    return merged_centers


def avg_cluster(local_center, weight):
    # 将 local_center 转换为 NumPy 数组
    local_center_array = np.array(local_center)

    # 将 weight 转换为 NumPy 数组，并归一化
    weight_array = np.array(weight)
    normalized_weights = weight_array / np.sum(weight_array)

    # 计算加权平均
    temp_center = np.average(local_center_array, axis=0, weights=normalized_weights)
    return temp_center


def kmeans_cluster(local_center, global_center):
    local_center_array = np.concatenate(local_center, axis=0)
    # 定义K-means聚类器，将数据聚类成10类
    num_clusters = 10
    # kmeans = KMeans(num_clusters).fit(local_center_array)
    if global_center is None or not global_center.any():
        kmeans = KMeans(num_clusters).fit(local_center_array)
    else:
        kmeans = KMeans(num_clusters, init=global_center).fit(local_center_array)
    # 使用K-means进行聚类
    kmeans.fit(local_center_array)

    # 获取聚类的标签和聚类中心
    temp_center = kmeans.cluster_centers_
    return temp_center

