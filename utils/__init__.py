from .nn import AutoEncoder, DEC, get_p, CAE
from .for_train import pretrain, train, get_initial_center, set_data_plot, plot, average_global_weights, avg_cluster, kmeans_cluster
from .for_eval import accuracy
from .data import load_mnist, load_emnist, load_usps, load_medical, load_fashion, load_cifar, load_eye, load_xray, load_reuters