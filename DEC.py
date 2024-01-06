import argparse
import copy
import os
import shutil
from pathlib import Path as p
from time import time
import numpy as np
import random

import torch
from matplotlib import pyplot as plt
from torch.nn import Parameter
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import (average_global_weights, load_mnist, set_data_plot, plot,
                   AutoEncoder, DEC, CAE,
                   pretrain, train, get_initial_center,
                   accuracy, load_usps, load_emnist, load_medical, load_fashion, load_cifar, load_eye, load_xray)


def get_arg():
    arg = argparse.ArgumentParser()
    arg.add_argument('-bs', default=256, type=int, help='batch size')
    arg.add_argument('-pre_epoch', default=200, type=int, help='epochs for train Autoencoder')
    arg.add_argument('-epoch', default=200, type=int, help='epochs for train DEC')
    arg.add_argument('-k', default=10, type=int, help='num of clusters')
    arg.add_argument('-save_dir', default='weight', help='location where model will be saved')
    arg.add_argument('-seed', default=10, type=int, help='torch random seed')
    arg.add_argument('-clients', default=1, type=int, help='torch random seed')
    arg = arg.parse_args()
    return arg


def main():
    arg = get_arg()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    # tr_ds, test_ds, val_ds = load_mnist(arg.bs, arg.clients)
    # tr_ds, test_ds = load_fashion(arg.bs, arg.clients)
    # tr_ds, test_ds = load_usps(arg.bs, arg.clients)
    #  tr_ds, test_ds = load_emnist(arg.bs, arg.clients)
    # tr_ds, test_ds = load_cifar(arg.bs, arg.clients)
    # tr_ds, test_ds = load_eye(arg.bs, arg.clients)
    # tr_ds, test_ds = load_xray(arg.bs, arg.clients)
    # tr_ds, test_ds = load_medical(arg.bs, arg.clients)
    global_adjusted_rand_idx, global_nmi, global_acc, client_acc, dec_acc = [], [], [], [], []
    global_center = np.zeros((10, 10))
    epsilon = 1.0
    delta = 1e-5
    max_grad_norm = 1.0

    for c in range(arg.clients):
        # tr_ds, test_ds, val_ds = load_mnist(arg.bs, arg.clients)
        tr_ds, test_ds = load_fashion(arg.bs, arg.clients)
        # tr_ds, test_ds = load_usps(arg.bs, arg.clients)
        # tr_ds, test_ds = load_medical(arg.bs, arg.clients)
        # set_data_plot(tr_ds[0], test_ds, device)
        print('train len: ', len(tr_ds[0].dataset))
        ae = AutoEncoder().to(device)
        print(f'\nAE param: {sum(p.numel() for p in ae.parameters()) / 1e6:.2f} M')
        opt = AdamW(ae.parameters())
        ae_dict = pretrain(ae, opt, tr_ds[0], device, arg.pre_epoch, arg.save_dir, 1)
        ae.load_state_dict(ae_dict)
        torch.save(ae, f'{arg.save_dir}/fine_tune_AE.pt')


        # initial center
        ae = torch.load(f'{arg.save_dir}/fine_tune_AE.pt', device)
        center= get_initial_center(ae, tr_ds[0], device, arg.k, arg.save_dir, 1, None)
        center = Parameter(torch.tensor(center,
                                            device=device,
                                            dtype=torch.float))

        # train dec
        print('\nload the best encoder and build DEC ...')
        ae = torch.load(f'{arg.save_dir}/fine_tune_AE.pt', device)
        dec = DEC(ae.encoder, center, alpha=1).to(device)
        print(f'DEC param: {sum(p.numel() for p in dec.parameters()) / 1e6:.2f} M')
        opt = SGD(dec.parameters(), 0.01, 0.9, nesterov=True)
        t4 = time()

        acc, nmi, adjusted_rand_idx = train(dec, opt, test_ds, device, arg.epoch, arg.save_dir, c)
        # plot(dec, arg.save_dir, 'train', c)
        global_acc.append(acc)
        global_nmi.append(nmi)
        global_adjusted_rand_idx.append(adjusted_rand_idx)

    plt.clf
    plt.figure()
    plt.title(f'global_acc')
    plt.plot(range(len(global_acc)), global_acc, color='r')
    plt.ylabel('Training acc')
    plt.xlabel('client')
    plt.savefig(f'{arg.save_dir}/acc_global_acc.png')
    plt.close()
    print("avg-acc: ", sum(global_acc)/len(global_acc))
    plt.clf
    plt.figure()
    plt.title(f'global_nmi')
    plt.plot(range(len(global_nmi)), global_nmi, color='r')
    plt.ylabel('Training nmi')
    plt.xlabel('client')
    plt.savefig(f'{arg.save_dir}/acc_global_nmi.png')
    plt.close()
    print("avg-nmi: ", sum(global_nmi)/len(global_nmi))

    plt.clf
    plt.figure()
    plt.title(f'global_adjusted_rand_idx')
    plt.plot(range(len(global_adjusted_rand_idx)), global_adjusted_rand_idx, color='r')
    plt.ylabel('Training adjusted_rand_idx')
    plt.xlabel('client')
    plt.savefig(f'{arg.save_dir}/acc_global_adjusted_rand_idx.png')
    plt.close()
    print("avg-adjusted_rand_idx: ", sum(global_adjusted_rand_idx)/len(global_adjusted_rand_idx))


if __name__ == '__main__':
    main()

