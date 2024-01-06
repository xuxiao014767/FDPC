import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, use_act=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        if use_act:
            self.act = nn.ReLU()
        self.use_act = use_act
         
    def forward(self, x):
        x = self.fc(x)
        if self.use_act:
            x = self.act(x) 
        return x
    

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, use_act=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        if use_act:
            self.act = nn.ReLU()
        self.use_act = use_act
         
    def forward(self, x):
        x = self.fc(x)
        if self.use_act:
            x = self.act(x) 
        return x
    

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(Encoder(784, 500, True),
                                     Encoder(500, 500, True),
                                     Encoder(500, 2000, True),
                                     Encoder(2000, 10, False))
        self.decoder = nn.Sequential(Decoder(10, 2000, True),
                                     Decoder(2000, 500, True),
                                     Decoder(500, 500, True),
                                     Decoder(500, 784, False))
            
    def forward(self, x):
        x  = self.encoder(x)
        gen = self.decoder(x)
        return x, gen


class Cluster(nn.Module):
    def __init__(self, center, alpha):
        super().__init__()
        self.center = center
        self.alpha = alpha

    def forward(self, x):
        square_dist = torch.pow(x[:, None, :] - self.center, 2).sum(dim=2)
        nom = torch.pow(1 + square_dist / self.alpha, -(self.alpha + 1) / 2)
        denom = nom.sum(dim=1, keepdim=True)
        return nom / denom


def get_p(q):
    with torch.no_grad():
        f = q.sum(dim=0, keepdim=True)
        nom = q ** 2 / f
        denom = nom.sum(dim=1, keepdim=True)
    return nom / denom
    
    
class DEC(nn.Module):
    def __init__(self, encoder, center, alpha=1):
        super().__init__()
        self.encoder = encoder
        self.cluster = Cluster(center, alpha)

    def forward(self, x):
        x = self.encoder(x)
        x = self.cluster(x)
        return x


class CAE(nn.Module):
    def __init__(self, input_shape=(1, 16, 16), filters=[32, 64, 128, 10]):
        super(CAE, self).__init__()

        if input_shape[1] % 8 == 0:
            pad3 = 1
        else:
            pad3 = 0

        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], filters[0], kernel_size=5, stride=2, padding=2), #(1,32)
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[1], kernel_size=5, stride=2, padding=2),     #(32,64)
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[1], filters[2], kernel_size=3, stride=2, padding=pad3),  #(64,128)
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(filters[2] * (input_shape[1] // 8) * (input_shape[2] // 8), filters[3]), #(512,10)
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(filters[3], filters[2] * (input_shape[1] // 8) * (input_shape[2] // 8)), #(10,512)
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (filters[2], input_shape[1] // 8, input_shape[2] // 8)),           #(1,(128,2,2))
            nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, stride=2, padding=pad3, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(filters[1], filters[0], kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(filters[0], input_shape[0], kernel_size=5, stride=2, padding=2, output_padding=1)
        )

    def forward(self, x):
        encodered = self.encoder(x)
        decodered = self.decoder(encodered)
        for name, layer in self.decoder.named_children():
            decoder_output = layer(decoder_output)
            print(f"Layer: {name}, Output Shape: {decoder_output.shape}")

        return encodered, decodered