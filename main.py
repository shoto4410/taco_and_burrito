import torch
from torch import nn, optim
from torch.utils.data import (Dataset, DataLoader, TensorDataset)
import tqdm

from torchvision.datasets import ImageFolder
from torchvision import transforms

train_imgs = ImageFolder(
    "taco_and_burrito/train",
    transform=transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor()
    ])
)
test_imgs = ImageFolder(
    "taco_and_burrito/test",
    transform=transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor()
    ])
)

#DataLoaderを作成
train_loader = DataLoader(
    train_imgs, batch_size=32, shuffle=True
)
test_loader = DataLoader(
    test_imgs, batch_size=32, shuffle=False
)

print(train_imgs.classes)
print(train_imgs.class_to_idx)


from torchvision import models
#resnet18 load
net = models.resnet18(pretrained=True)

#all parameters 微分対象外にする
for p in net.parameters():
    p.requires_grad = False

#最後の線形層を付け替える
fc_input_dim = net.fc.in_features
net.fc = nn.Linear(fc_input_dim, 2)

def eval_net(net, data_loader, device="cpu"):
    #DropoutやBatchnormを無効化
    net.eval()
    ypreds = []
    ys = []
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        #確率が最大のクラスを予測
        with torch.no_grad():
            _, y_pred = net(x).max(1)
        ys.append(y)
        ypreds.append(y_pred)
        #予測精度を計算
        acc = (ys == ypreds).float().sum() / len(ys)
        return acc.item()