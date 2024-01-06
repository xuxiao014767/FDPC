import os

from torch import nn
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import random_split

class MyDataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(MyDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.samples = self._make_dataset(root)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.samples)

    def _make_dataset(self, dir):
        samples = []
        label_names = os.listdir(dir)
        for label_name in label_names:
            label_dir = os.path.join(dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            for file_name in os.listdir(label_dir):
                if not file_name.endswith('.jpg') and not file_name.endswith('.png'):
                    continue
                path = os.path.join(label_dir, file_name)
                item = (path, int(label_name))
                samples.append(item)
        return samples

t = T.Compose([T.Resize((224, 224)),
                T.ToTensor()],
                nn.Flatten(0))

mydataset = MyDataset(os.getcwd(), transform=t)

# 划分数据集
train_size = int(0.8 * len(mydataset))  # 划分训练集和验证集的比例为8:2
valid_size = len(mydataset) - train_size
train_dataset, valid_dataset = random_split(mydataset, [train_size, valid_size])

# 定义 DataLoader
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=True)
