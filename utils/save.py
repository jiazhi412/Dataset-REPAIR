import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_pkl(pkl_data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(pkl_data, f)

def get_feature(train_loader, test_loaders, color_fn, opt):
    epochs = opt['epochs']
    lr = opt['lr']
    device = opt['device']
    model = create_mnist_model(opt['model'])
    # create models
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.1)

    # training
    train(train_loader, epochs, model, optimizer, scheduler)

    # testing
    with torch.no_grad():
        corrects = 0

        for x, y in test_loaders:
            x, y = x.to(device), y.to(device)
            colors = color_fn(x)
            out, feature = model(x)
            corrects += out.max(1)[1].eq(y).sum().item()

        acc = corrects / len(test_loaders.dataset)
        print('Test accuracy on Colored MNIST = {:.2%}'.format(acc))
    return feature, y, colors

def train(loader, epochs, model, optimizer, scheduler=None):
    model.train()
    with tqdm(range(1, epochs + 1)) as pbar:
        for _ in pbar:
            losses = []
            corrects = 0
            if scheduler is not None:
                scheduler.step()

            for x, y in loader:
                out, feature = model(x)
                loss = F.cross_entropy(out, y)
                losses.append(loss.item())
                corrects += out.max(1)[1].eq(y).sum().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss = sum(losses) / len(losses)
            acc = 100 * corrects / len(loader.dataset)
            pbar.set_postfix(loss='%.3f' % loss, acc='%.2f%%' % acc)
    return loss, acc



class LeNet(nn.Module):
    def __init__(self, in_channels, out_dims):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, out_dims)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x), x


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            in_dim = dim
        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        if len(x.size()) > 2:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return self.out(x)


def create_mnist_model(model_name):
    if model_name == 'lenet':
        return LeNet(3, 10)
    elif model_name == 'mlp':
        return MLP(784 * 3, [300, 100], 10)
    else:
        raise ValueError('Model not supported')