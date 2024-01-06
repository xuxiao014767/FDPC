from torchvision.datasets import VisionDataset
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST, USPS, EMNIST, FashionMNIST, CIFAR10
from torchvision import transforms as T, datasets
import numpy as np
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
import random
from collections import Counter
from keras.datasets import reuters
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def default_loader(path):
    return Image.open(path).convert('L')



def split_iid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    idxs = np.arange(len(dataset))
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)
    # 创建一个 defaultdict，键为标签，值为所有索引组成的列表
    idxs_dict = defaultdict(list)
    for i, label in enumerate(labels):
        idxs_dict[label].append(idxs[i])

    all_idxs = [i for i in idxs_dict.values()]  # 用户字典， 数据集dataset的索引
    num = [int(len(i) / num_users) for i in all_idxs]
    dict_users = {}
    for i in range(num_users):
        dict_users[i] = set()
    for x in range(len(all_idxs)):#10
        for y in range(len(dict_users)):#50
            if num[x] < len(all_idxs[x]):
                temp = set(np.random.choice(all_idxs[x], num[x], replace=False))
            else:
                temp = set(all_idxs[x])
            dict_users[y].update(temp)
            all_idxs[x] = list(set(all_idxs[x]) - temp)
    return dict_users  # 返回了每个用户以及所对应的600个数据的字典。



def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    参数为 alpha 的 Dirichlet 分布将数据索引划分为 n_clients 个子集
    '''
    # 总类别数
    n_classes = train_labels.max()+1

    # [alpha]*n_clients 如下：
    # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # 得到 62 * 10 的标签分布矩阵，记录每个 client 占有每个类别的比率
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    # 记录每个类别对应的样本下标
    # 返回二维数组
    class_idcs = [np.argwhere(train_labels==y).flatten()
           for y in range(n_classes)]

    # 定义一个空列表作最后的返回值
    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [set(np.concatenate(idcs)) for idcs in client_idcs]
    # 创建一个字典，键为0到9，值为对应的NumPy数组
    result_dict = {key: client_idcs[key] for key in range(n_clients)}
    return result_dict

def mixture_distribution_split_noniid(dataset, n_classes, n_clients, n_clusters, alpha, seed):
    if n_clusters == -1:
        n_clusters = n_classes
        
    all_labels = list(range(n_classes))
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.shuffle(all_labels)

    def avg_divide(l, g):
        num_elems = len(l)
        group_size = int(len(l) / g)
        num_big_groups = num_elems - g * group_size
        num_small_groups = g - num_big_groups
        glist = []
        for i in range(num_small_groups):
            glist.append(l[group_size * i: group_size * (i + 1)])
        bi = group_size * num_small_groups
        group_size += 1
        for i in range(num_big_groups):
            glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
        return glist
    
    clusters_labels = avg_divide(all_labels, n_clusters)

    label2cluster = dict()  
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels:
            label2cluster[label] = group_idx
            
    data_idcs = list(range(len(dataset)))
    clusters_sizes = np.zeros(n_clusters, dtype=int)
    clusters = {k: [] for k in range(n_clusters)}
    for idx in data_idcs:
        _, label = dataset[idx]
        group_id = label2cluster[label]
        clusters_sizes[group_id] += 1
        clusters[group_id].append(idx)
    for _, cluster in clusters.items():
        rng.shuffle(cluster)
    
    clients_counts = np.zeros((n_clusters, n_clients), dtype=np.int64) 
    for cluster_id in range(n_clusters):
        weights = np.random.dirichlet(alpha=alpha * np.ones(n_clients))
        clients_counts[cluster_id] = np.random.multinomial(clusters_sizes[cluster_id], weights)
    clients_counts = np.cumsum(clients_counts, axis=1)

    def split_list_by_idcs(l, idcs):
        res = []
        current_index = 0
        for index in idcs: 
            res.append(l[current_index: index])
            current_index = index
        return res
    
    clients_idcs = [[] for _ in range(n_clients)]
    for cluster_id in range(n_clusters):
        cluster_split = split_list_by_idcs(clusters[cluster_id], clients_counts[cluster_id])
        for client_id, idcs in enumerate(cluster_split):
            clients_idcs[client_id] += idcs
    result_dict = {key: clients_idcs[key] for key in range(n_clients)}        
    return result_dict


def load_mnist(batch_size, clients):
    t = T.Compose([T.ToTensor(),
                   T.Normalize((0.1307,), (0.3081,)),
                   nn.Flatten(0)])

    # tr_ds = MNIST('mnist', True, t, download=True)
    n_classes = 10
    n_components = 3
    dirichlet_alpha = 1.0
    seed = 42
    # 定义要获取的数据集大小
    num_samples = 10000
    end_samples = 12000

    val_ds = MNIST('mnist', True, transform=t, download=True)
    val_ds.data = val_ds.data[num_samples:end_samples]
    val_ds.targets = val_ds.targets[num_samples:end_samples]
    val_labels = np.array(val_ds.targets)
    # 从MNIST数据集中获取前num_samples个样本
    tr_ds = MNIST('mnist', True, transform=t, download=True)
    print(tr_ds.data.shape)
    tr_ds.data = tr_ds.data[40000:50000]
    tr_ds.targets = tr_ds.targets[40000:50000]
    train_labels = np.array(tr_ds.targets)

    # 从tr_ds中随机选择2000个样本作为验证集test_ds
    random_indices = random.sample(range(len(tr_ds)), 2000)
    test_ds = torch.utils.data.Subset(tr_ds, random_indices)
    test_labels = train_labels[random_indices]
    print(tr_ds.data.shape)
    #user_groups = split_iid(tr_ds, clients)  # 返回c个预分配索引
    user_groups = dirichlet_split_noniid(train_labels, 1.0, clients)  # 返回c个预分配索引
    #user_groups = mixture_distribution_split_noniid(tr_ds, n_classes, clients, n_components, dirichlet_alpha, seed)

    # 查看数据集分布图
    plt.rc('font', size=15)
    plt.figure(figsize=(14, 5))  # Increase the figure size for better readability
    plt.hist([train_labels[np.array(list(idc))] for idc in user_groups.values()], stacked=True,
            bins=np.arange(min(train_labels) - 0.5, max(train_labels) + 1.5, 1),
            label=["Client {}".format(i) for i in range(clients)], rwidth=0.5)

    plt.xticks(np.arange(10), tr_ds.classes)

    # Adjust legend position to upper right and add some padding
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, fontsize=13)
    plt.xlim(-0.5, 11)
    plt.ylim(0, 1200)
    # Save and show the plot
    plt.savefig('./weight/mnist_seg.png', bbox_inches='tight')  # Use bbox_inches to include legend

    tr_ds = [(x, torch.tensor(y)) for x, y in tr_ds]
    # test_ds = MNIST('mnist', False, t, download=True)
    # test_ds.data = test_ds.data[:2000]
    # test_ds.targets = test_ds.targets[:2000]
    test_ds = [(x, torch.tensor(y)) for x, y in test_ds]

    trainloader = []
    for idx in user_groups.values():
        idx = list(idx)
        trainloader.append(DataLoader(DatasetSplit(tr_ds, idx),
                                      batch_size=batch_size, shuffle=True, num_workers=0))

    test_dataset = DataLoader(test_ds,
                              batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataset = DataLoader(val_ds,
                              batch_size=batch_size, shuffle=True, num_workers=0)
    return trainloader, test_dataset, val_dataset



class DatasetSplit(Dataset): #使用dataset重构
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):   # 返回数据列表长度，即数据集的样本数量
        return len(self.idxs)

    def __getitem__(self, item): # 通过dataset读取图像数据，最后返回下标为item的图像数据和标签的张量。
        image, label = self.dataset[self.idxs[item]]
        return torch.as_tensor(image), torch.as_tensor(label)  #torch.tensor() #转换为张量形式，且会拷贝data


def load_reuters(bs,c):
    class ReutersDataset(Dataset):
        def __init__(self, data, targets):
            self.data = data
            self.targets = targets

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    
    # 加载reuters数据集
    (train_data, train_labels), (test_data, test_labels) = reuters.load_mnist(num_words=10000)
    # Define the labels to keep
    labels_to_keep = [3, 4, 16, 19]

    # Create a mask to filter the data based on the specified labels
    train_mask = np.isin(train_labels, labels_to_keep)
    test_mask = np.isin(test_labels, labels_to_keep)

    # Apply the mask to filter the data and labels
    train_data = train_data[train_mask]
    train_labels = train_labels[train_mask]
    test_data = test_data[test_mask]
    test_labels = test_labels[test_mask]

    def vectorize_sequences(sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results


    # 将训练数据和测试数据向量化
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    tr_ds = ReutersDataset(x_train, train_labels)
    test_ds = ReutersDataset(x_test, test_labels)

    class_labels = tr_ds.targets
    # Count the number of samples for each class
    class_counts = dict(Counter(class_labels))

    # Sort the classes and corresponding counts
    sorted_classes = sorted(class_counts.keys())
    # user_groups = split_iid(tr_ds, c)  # 返回c个预分配索引
    user_groups = dirichlet_split_noniid(train_labels, 1.0, c)  # 返回c个预分配索引
    # user_groups = mixture_distribution_split_noniid(tr_ds, n_classes, clients, n_components, dirichlet_alpha, seed)
    x = train_labels
    mapping = {3: 0, 4: 1, 16: 2, 19: 3}
    # 使用映射关系替换标签
    for i in range(len(x)):
        x[i] = mapping.get(x[i], x[i])
    # 查看数据集分布图
    plt.clf()
    plt.rc('font', size=15)
    plt.figure(figsize=(13, 5))  # Increase the figure size for better readability
    plt.hist([x[np.array(list(idc))] for idc in user_groups.values()], stacked=True,
            bins=np.arange(min(x) - 0.5, max(x) + 1.5, 1),
            label=["Client {}".format(i) for i in range(c)], rwidth=0.5)

    plt.xticks(np.arange(4), ['?','the','it','1'])

    # Adjust legend position to upper right and add some padding
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, fontsize=13)
    # Save and show the plot
    plt.savefig('./weight/reuters_seg.png', bbox_inches='tight')  # Use bbox_inches to include legend


    tr_ds = [(x, torch.tensor(y)) for x, y in tr_ds]
    test_ds = [(x, torch.tensor(y)) for x, y in test_ds]

    trainloader = []
    for idx in user_groups.values():
        idx = list(idx)
        trainloader.append(DataLoader(DatasetSplit(tr_ds, idx),
                                    batch_size=bs, shuffle=True, num_workers=0))

    test_dataset = DataLoader(test_ds,
                            batch_size=bs, shuffle=True, num_workers=0)
    return trainloader, test_dataset


def load_usps(bs, c):
    t = T.Compose([T.ToTensor(),
                   T.Normalize((0.1307,), (0.3081,)),
                   nn.Flatten(0)])
    # num_samples = 2000
    tr_ds = USPS('mnist/USPS', True, t)
    # tr_ds.data = tr_ds.data[:num_samples]
    # tr_ds.targets = tr_ds.targets[:num_samples]
    # Assuming `tr_ds.targets` contains the class labels
    class_labels = tr_ds.targets
    # Count the number of samples for each class
    class_counts = dict(Counter(class_labels))

    # Sort the classes and corresponding counts
    sorted_classes = sorted(class_counts.keys())
    sample_counts = [class_counts[label] for label in sorted_classes]
    n_classes = 10
    n_components = 3
    dirichlet_alpha = 1.0
    seed = 42
    train_labels = np.array(tr_ds.targets)
    train_labels = np.array(tr_ds.targets)
    #user_groups = split_iid(tr_ds, c)  # 返回c个预分配索引
    user_groups = dirichlet_split_noniid(train_labels, 1.0, c)  # 返回c个预分配索引
    #user_groups = mixture_distribution_split_noniid(tr_ds, n_classes, c, n_components, dirichlet_alpha, seed)
    # Create a bar plot
    plt.rc('font', size=15)
    plt.figure(figsize=(13, 5))
    plt.hist([train_labels[np.array(list(idc))] for idc in user_groups.values()], stacked=True,
             bins=np.arange(min(train_labels) - 0.5, max(train_labels) + 1.5, 1),
             label=["Client {}".format(i) for i in range(c)], rwidth=0.5)
    plt.xticks(np.arange(10), sorted_classes)
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, fontsize=13)
    plt.xlim(-0.5, 11)
    plt.savefig('./weight/usps_seg.png')

    tr_ds = [(x, torch.tensor(y)) for x, y in tr_ds]
    random_indices = random.sample(range(len(tr_ds)), 2000)
    test_ds = torch.utils.data.Subset(tr_ds, random_indices)
    test_labels = train_labels[random_indices]
    test_ds = [(x, torch.tensor(y)) for x, y in test_ds]

    trainloader = []
    for idx in user_groups.values():
        idx = list(idx)
        trainloader.append(DataLoader(DatasetSplit(tr_ds, idx),
                                      batch_size=bs, shuffle=True, num_workers=0))

    test_dataset = DataLoader(test_ds,
                              batch_size=bs, shuffle=True, num_workers=0)

    return trainloader, test_dataset


def load_emnist(bs, c):
    t = T.Compose([T.ToTensor(),
                   T.Normalize((0.1307,), (0.3081,)),
                   nn.Flatten(0)])

    tr_ds = EMNIST(root='mnist', split='digits', train=True, transform=t, download=True)
    user_groups = split_iid(tr_ds, c)  # 返回c个预分配索引
    tr_ds = [(x, torch.tensor(y)) for x, y in tr_ds]
    test_ds = EMNIST('mnist', split='digits', train=False, download=True,transform=t)
    test_ds = [(x, torch.tensor(y)) for x, y in test_ds]


    trainloader = []
    for idx in user_groups.values():
        idx = list(idx)
        trainloader.append(DataLoader(DatasetSplit(tr_ds, idx),
                                      batch_size=bs, shuffle=True, num_workers=0))

    test_dataset = DataLoader(test_ds,
                              batch_size=bs, shuffle=True, num_workers=0)

    return trainloader, test_dataset


def load_fashion(batch_size, clients):
    t = T.Compose([T.ToTensor(),
                   T.Normalize((0.1307,), (0.3081,)),
                   nn.Flatten(0)])
        # 定义要获取的数据集大小
    num_samples = 10000

    # 从MNIST数据集中获取前num_samples个样本
    tr_ds = FashionMNIST('mnist', True, transform=t, download=True)
    tr_ds.data = tr_ds.data[:num_samples]
    tr_ds.targets = tr_ds.targets[:num_samples]
    train_labels = np.array(tr_ds.targets)

    # user_groups = split_iid(tr_ds, clients)  # 返回c个预分配索引
    user_groups = dirichlet_split_noniid(train_labels, 1.0, clients)  # 返回c个预分配索引
    # 查看数据集分布图
    plt.rc('font', size=15)
    plt.figure(figsize=(14, 5))
    plt.hist([train_labels[np.array(list(idc))] for idc in user_groups.values()], stacked=True,
             bins=np.arange(min(train_labels) - 0.5, max(train_labels) + 1.5, 1),
             label=["Client {}".format(i) for i in range(clients)], rwidth=0.5)
    plt.xticks(np.arange(10), tr_ds.classes)
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, fontsize=13)
    plt.ylim(0, 1100)
    plt.xlim(-0.5, 11)
    plt.savefig('./weight/fashion_seg.png')

    tr_ds = [(x, torch.tensor(y)) for x, y in tr_ds]
    random_indices = random.sample(range(len(tr_ds)), 2000)
    test_ds = torch.utils.data.Subset(tr_ds, random_indices)
    test_labels = train_labels[random_indices]
    test_ds = [(x, torch.tensor(y)) for x, y in test_ds]


    trainloader = []
    for idx in user_groups.values():
        idx = list(idx)
        trainloader.append(DataLoader(DatasetSplit(tr_ds, idx),
                                      batch_size=batch_size, shuffle=True, num_workers=0))

    test_dataset = DataLoader(test_ds,
                              batch_size=batch_size, shuffle=True, num_workers=0)

    return trainloader, test_dataset


def load_cifar(batch_size, clients):
    t = T.Compose([T.ToTensor(),
                   T.Normalize((0.1307,), (0.3081,)),
                   nn.Flatten(0)])

    tr_ds = CIFAR10('mnist', True, t, download=True)
    user_groups = split_iid(tr_ds, clients)  # 返回c个预分配索引
    tr_ds = [(x, torch.tensor(y)) for x, y in tr_ds]
    test_ds = CIFAR10('mnist', False, t, download=True)
    test_ds = [(x, torch.tensor(y)) for x, y in test_ds]

    trainloader = []
    for idx in user_groups.values():
        idx = list(idx)
        trainloader.append(DataLoader(DatasetSplit(tr_ds, idx),
                                      batch_size=batch_size, shuffle=True, num_workers=0))

    test_dataset = DataLoader(test_ds,
                              batch_size=batch_size, shuffle=True, num_workers=0)

    return trainloader, test_dataset

def load_eye(bs, c):
    class MyDataset(VisionDataset):
        def __init__(self, root, transform=None, target_transform=None):
            super(MyDataset, self).__init__(root, transform=transform, target_transform=target_transform)
            self.samples, self.targets = self._make_dataset(root)

        def __getitem__(self, index):
            path = self.samples[index]
            target = self.targets[index]
            with open(path, 'rb') as f:
                img = Image.open(f).convert('L')
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

        def __len__(self):
            return len(self.samples)

        def _make_dataset(self, dir):
            samples = []
            targets = []
            label_names = os.listdir(dir)
            for label_name in label_names:
                label_dir = os.path.join(dir, label_name)
                if not os.path.isdir(label_dir):
                    continue
                for file_name in os.listdir(label_dir):
                    if not file_name.endswith('.tif') and not file_name.endswith('.png'):
                        continue
                    path = os.path.join(label_dir, file_name)
                    samples.append(path)
                    targets.append(int(label_name))
            return samples, targets

    t = T.Compose([T.ToTensor(),
                   T.Resize((16, 16)),
                   T.Normalize((0.1307,), (0.3081,)),
                   nn.Flatten(0)])

    mydataset = MyDataset(os.path.join('mnist/eyes/Base11'), transform=t)
    class_labels = mydataset.targets
    # Count the number of samples for each class
    class_counts = dict(Counter(class_labels))

    # Sort the classes and corresponding counts
    sorted_classes = sorted(class_counts.keys())
    sample_counts = [class_counts[label] for label in sorted_classes]
    plt.figure(figsize=(4, 6))
    plt.bar(sorted_classes, sample_counts)
    plt.xlabel('Class Label')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of USPS Training Samples by Class')
    plt.xticks(sorted_classes)
    plt.savefig('weight/Distribution of USPS Training Samples by Class.png')

    tr_ds = [(x, torch.tensor(y)) for x, y in mydataset]
    user_groups = split_iid(mydataset, c)  # 返回c个预分配索引
    testloader = DataLoader(tr_ds,
               batch_size=bs, shuffle=True, num_workers=0)

    trainloader = []
    for idx in user_groups.values():
        idx = list(idx)
        trainloader.append(DataLoader(DatasetSplit(tr_ds, idx),
                                      batch_size=bs, shuffle=True, num_workers=0))

    return trainloader, testloader

def load_medical(bs, c):
    class MyDataset(VisionDataset):
        def __init__(self, root, transform=None, target_transform=None):
            super(MyDataset, self).__init__(root, transform=transform, target_transform=target_transform)
            self.samples, self.targets = self._make_dataset(root)

        def __getitem__(self, index):
            path = self.samples[index]
            target = self.targets[index]
            with open(path, 'rb') as f:
                img = Image.open(f).convert('L')
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

        def __len__(self):
            return len(self.samples)

        def _make_dataset(self, dir):
            samples = []
            targets = []
            label_names = os.listdir(dir)
            for label_name in label_names:
                label_dir = os.path.join(dir, label_name)
                if not os.path.isdir(label_dir):
                    continue
                for file_name in os.listdir(label_dir):
                    if not file_name.endswith('.tif') and not file_name.endswith('.png'):
                        continue
                    path = os.path.join(label_dir, file_name)
                    samples.append(path)
                    targets.append(int(label_name))
            return samples, targets

    t = T.Compose([T.ToTensor(),
                   T.Resize((9, 9)),
                   T.Normalize((0.1307,), (0.3081,)),
                   nn.Flatten(0)])

    mydataset = MyDataset(os.path.join('mnist/Dataset_BUSI_with_GT'), transform=t)    

    train_labels = np.array(mydataset.targets)
    user_groups = split_iid(mydataset, c)  # 返回c个预分配索引
    # user_groups = dirichlet_split_noniid(train_labels, 1.0, c)  # 返回c个预分配索引
    tr_ds = [(x, torch.tensor(y)) for x, y in mydataset]
    random_indices = random.sample(range(len(tr_ds)), 320)
    test_ds = torch.utils.data.Subset(tr_ds, random_indices)
    test_labels = train_labels[random_indices]


    testloader = DataLoader(test_ds,
               batch_size=bs, shuffle=True, num_workers=0)

    trainloader = []
    for idx in user_groups.values():
        idx = list(idx)
        trainloader.append(DataLoader(DatasetSplit(tr_ds, idx),
                                      batch_size=bs, shuffle=True, num_workers=0))

    return trainloader, testloader


def load_xray(bs, c):
    class MyDataset(VisionDataset):
        def __init__(self, root, transform=None, target_transform=None):
            super(MyDataset, self).__init__(root, transform=transform, target_transform=target_transform)
            self.samples, self.targets = self._make_dataset(root)

        def __getitem__(self, index):
            path = self.samples[index]
            target = self.targets[index]
            with open(path, 'rb') as f:
                img = Image.open(f).convert('L')
                img = Image.merge('RGB', (img, img, img))  # Convert to RGB by replicating the single channel
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

        def __len__(self):
            return len(self.samples)

        def _make_dataset(self, dir):
            samples = []
            targets = []
            label_names = os.listdir(dir)
            for label_name in label_names:
                label_dir = os.path.join(dir, label_name)
                if not os.path.isdir(label_dir):
                    continue
                for file_name in os.listdir(label_dir):
                    if not file_name.endswith('.tif') and not file_name.endswith('.jpeg'):
                        continue
                    path = os.path.join(label_dir, file_name)
                    samples.append(path)
                    targets.append(int(label_name))
            return samples, targets

    t = T.Compose([
        T.Resize(300),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],# mean
                    [0.229, 0.224, 0.225]),# std 
        nn.Flatten(0)])

    mydataset = MyDataset(os.path.join('mnist/Xray'), transform=t)    
    class_labels = mydataset.targets
    # Count the number of samples for each class
    class_counts = dict(Counter(class_labels))

    # Sort the classes and corresponding counts
    sorted_classes = sorted(class_counts.keys())
    sample_counts = [class_counts[label] for label in sorted_classes]
    plt.figure(figsize=(4, 6))
    plt.bar(sorted_classes, sample_counts)
    plt.xlabel('Class Label')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Xray Training Samples by Class')
    plt.xticks(sorted_classes)
    plt.savefig('weight/Distribution of Xray Training Samples by Class.png')

    train_labels = np.array(mydataset.targets)
    # user_groups = split_iid(mydataset, c)  # 返回c个预分配索引
    user_groups = dirichlet_split_noniid(train_labels, 1.0, c)  # 返回c个预分配索引
    tr_ds = [(x, torch.tensor(y)) for x, y in mydataset]
    


    testloader = DataLoader(tr_ds,
               batch_size=bs, shuffle=True, num_workers=0)

    trainloader = []
    for idx in user_groups.values():
        idx = list(idx)
        trainloader.append(DataLoader(DatasetSplit(tr_ds, idx),
                                      batch_size=bs, shuffle=True, num_workers=0))

    return trainloader, testloader


# 首先继承上面的dataset类。然后在__init__()方法中得到图像的路径，然后将图像路径组成一个数组，这样在__getitim__()中就可以直接读取：
class MyDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()  # 对继承自父类的属性进行初始化
        fh = open(txt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []
        for line in fh:  # 迭代该列表#按行循环txt文本中的内
            line = line.strip('\n')
            line = line.rstrip('\n')  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = self.loader(fn)  # 按照路径读取图片
        if self.transform is not None:
            img = self.transform(img)  # 数据标签转换为Tensor
        return img, label  # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)
