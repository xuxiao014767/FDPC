import argparse
import copy
import os
import shutil
from pathlib import Path as p
from time import time
import numpy as np
import random
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.nn import Parameter
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.cluster import KMeans
from utils import (average_global_weights, load_mnist, set_data_plot, plot,
                   AutoEncoder, DEC,
                   pretrain, train, get_initial_center, avg_cluster, kmeans_cluster,
                   accuracy, load_emnist, load_usps, load_xray, load_medical, load_fashion, load_eye)

def get_arg():
    arg = argparse.ArgumentParser()
    arg.add_argument('-bs', default=256, type=int, help='batch size')
    arg.add_argument('-pre_epoch', default=200, type=int, help='epochs for train Autoencoder')
    arg.add_argument('-epoch', default=50, type=int, help='epochs for train DEC')
    arg.add_argument('-k', default=10, type=int, help='num of clusters')
    arg.add_argument('-save_dir', default='weight', help='location where model will be saved')
    arg.add_argument('-seed', default=10, type=int, help='torch random seed')
    arg.add_argument('-clients', default=20, type=int, help='torch random seed')
    arg.add_argument('-mode', default='weighting', type=str, help='weighting or kmeans')
    arg.add_argument('-dataset', default='fashion', type=str, help='fashion,mnist,usps,medical')
    arg = arg.parse_args()
    return arg


def main():
    arg = get_arg()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(arg.save_dir):
        os.makedirs(arg.save_dir, exist_ok=True)
    else:
        for file_name in os.listdir(arg.save_dir):
            file_path = os.path.join(arg.save_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    if arg.seed is not None:
        random.seed(arg.seed)
        np.random.seed(arg.seed)
        torch.manual_seed(arg.seed)
        torch.cuda.manual_seed(arg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


    if arg.dataset == 'fashion':
        tr_ds, _ = load_fashion(arg.bs, arg.clients)
    elif arg.dataset == 'mnist':
        tr_ds, _, val_ds = load_mnist(arg.bs, arg.clients)
    elif arg.dataset == 'usps':
        tr_ds, _ = load_usps(arg.bs, arg.clients)
    else:
        tr_ds, _ = load_medical(arg.bs, arg.clients)
    convergence_threshold = 0.0001  # Set your desired convergence threshold here
    prev_acc = 0  # Variable to store the accuracy from the previous epoch
    global_adjusted_rand_idx, global_nmi, global_acc, dec_acc = [], [], [], []
    global_center = np.zeros((10, 10))
    ae = AutoEncoder().to(device)
    torch.save(ae, f'{arg.save_dir}/fine_tune_AE.pt')
    for epoch in range(arg.epoch):#50
        if arg.dataset == 'fashion':
            _, test_ds = load_fashion(arg.bs, arg.clients)
        elif arg.dataset == 'mnist':
            _, test_ds, _ = load_mnist(arg.bs, arg.clients)
        elif arg.dataset == 'usps':
            _, test_ds = load_usps(arg.bs, arg.clients)
        else:
            _, test_ds = load_medical(arg.bs, arg.clients)
        local_center, local_model, weight = [], [], []
        for c in range(arg.clients):#50
            weight.append(len(tr_ds[c].dataset))
            print('\ntrain num:', len(tr_ds[c].dataset))
            print('test num:', len(test_ds.dataset))
            set_data_plot(tr_ds[c], test_ds, device)
            # train autoencoder
            ae = torch.load(f'{arg.save_dir}/fine_tune_AE.pt', device)
            print(f'\nAE param: {sum(p.numel() for p in ae.parameters()) / 1e6:.2f} M')
            opt = AdamW(ae.parameters())
            ae_dict = pretrain(ae, opt, tr_ds[c], device, arg.pre_epoch, arg.save_dir, c) #50
            local_model.append(copy.deepcopy(ae_dict))

            # initial center
            ae.load_state_dict(ae_dict)
            if arg.mode == 'weighting':
                center= get_initial_center(ae, tr_ds[c], device, arg.k, arg.save_dir, c, global_center)
            else:
                center= get_initial_center(ae, tr_ds[c], device, arg.k, arg.save_dir, c, None)
            local_center.append(center)

        avg_ae = average_global_weights(local_model)
        ae.load_state_dict(avg_ae)
        torch.save(ae, f'{arg.save_dir}/fine_tune_AE.pt')

        if arg.mode == 'weighting':
            temp_center = avg_cluster(local_center, weight)
        else:
            temp_center = kmeans_cluster(local_center, global_center)

        global_center = temp_center
        dec_center = Parameter(torch.tensor(temp_center,
                                            device=device,
                                            dtype=torch.float))

        ae = torch.load(f'{arg.save_dir}/fine_tune_AE.pt', device)
        dec = DEC(ae.encoder, dec_center, alpha=1).to(device)
        print(f'DEC param: {sum(p.numel() for p in dec.parameters()) / 1e6:.2f} M')
        opt = SGD(dec.parameters(), 0.01, 0.9, nesterov=True)
        t4 = time()
        acc, nmi, adjusted_rand_idx = train(dec, opt, test_ds, device, 200, arg.save_dir, epoch)
        global_acc.append(acc)
        global_nmi.append(nmi)
        global_adjusted_rand_idx.append(adjusted_rand_idx)
        # if len(global_acc)>=2:
        #     prev_acc = global_acc[-2]
            # Check if accuracy has converged
        if abs(acc - prev_acc) <= convergence_threshold:
            print(f'Convergence reached. Stopping training at epoch {epoch}.')
            break
        print(f'test acc: {acc:.4f}')
        print('*' * 50)
        t5 = time()


    # 清除之前的图形
    plt.clf()
    plt.figure()

    # 绘制全局NMI曲线
    plt.title('DEC-federate Metrics')
    plt.plot(range(len(global_nmi)), global_nmi, label='global_nmi', color='r')
    plt.plot(range(len(global_acc)), global_acc, label='global_acc', color='g')
    plt.plot(range(len(global_adjusted_rand_idx)), global_adjusted_rand_idx, label='global_adjusted_rand_idx', color='b')

    plt.ylabel('Metrics Value')
    plt.xlabel('Epoch')
    plt.legend()

    # 保存图像
    plt.savefig(f'{arg.save_dir}/DEC-federate-metrics.png')
    plt.close()
    # 清除之前的图形
    plt.clf()
    plt.figure(figsize=(10, 6))  # 调整图形的大小

    # 绘制全局NMI曲线
    plt.title('DEC-federate Metrics', fontsize=16)  # 设置标题和字体大小
    plt.plot(range(len(global_nmi)), global_nmi, label='global_nmi', color='r', linestyle='-', marker='o', markersize=5, linewidth=2)
    plt.plot(range(len(global_acc)), global_acc, label='global_acc', color='g', linestyle='--', marker='s', markersize=5, linewidth=2)
    plt.plot(range(len(global_adjusted_rand_idx)), global_adjusted_rand_idx, label='global_adjusted_rand_idx', color='b', linestyle='-.', marker='^', markersize=5, linewidth=2)

    plt.ylabel('Metrics Value', fontsize=14)  # 设置纵坐标标签和字体大小
    plt.xlabel('Epoch', fontsize=14)  # 设置横坐标标签和字体大小
    plt.legend(fontsize=12)  # 设置图例字体大小
    plt.grid(True, linestyle='--', alpha=0.6)  # 添加网格线，并设置样式和透明度

    # 保存图像
    plt.savefig(f'{arg.save_dir}/DEC-federate-metrics1.png', dpi=300, bbox_inches='tight')  # 保存高分辨率图像，并确保边界不被裁剪
    plt.close()
    max_value = max(global_acc)
    max_index = global_acc.index(max_value)

    # 输出均值
    print("max-acc: ", max_value)
    print("max-nmi: ", global_nmi[max_index])
    print("max-adjusted_rand_idx: ", global_adjusted_rand_idx[max_index])
    print("lastest-acc: ", global_acc[-1])
    global_acc.append(max_value)
    global_nmi.append(global_nmi[max_index])
    global_adjusted_rand_idx.append(global_adjusted_rand_idx[max_index])
    df = pd.DataFrame(zip(range(1, arg.epoch+2), global_acc, global_nmi, global_adjusted_rand_idx), columns=['epoch', 'acc', 'nmi', 'ari'])
    df.to_excel(f'{arg.save_dir}/train_global.xlsx', index=False)
    # # 输出均值
    # print("avg-nmi: ", sum(global_nmi)/len(global_nmi))
    # print("avg-acc: ", sum(global_acc)/len(global_acc))
    # print("avg-adjusted_rand_idx: ", sum(global_adjusted_rand_idx)/len(global_adjusted_rand_idx))

if __name__ == '__main__':
    main()

