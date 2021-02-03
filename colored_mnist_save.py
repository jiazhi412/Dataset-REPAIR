import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

from utils.datasets import ColoredDataset
from utils.measure import *
from utils.models import *
from utils.save import *

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--with-cuda', dest='cuda', action='store_true')
parser.add_argument('--model', default='lenet', choices=['lenet', 'mlp'], type=str)
parser.add_argument('--color-std', type=float, default=1e-1)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float)
parser.add_argument('--epochs', default=20, type=int)
args = parser.parse_args()

opt = vars(parser.parse_args())
opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')

root = './data'
if not os.path.exists(root):
    os.mkdir(root)

# load data
train_set = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
test_set = datasets.MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())

# biased datasets, i.e. colored mnist
print('Coloring MNIST dataset with standard deviation = {:.2f}'.format(args.color_std))
colored_train_set = ColoredDataset(train_set, classes=10, colors=[0, 1], std=args.color_std)
train_loader = DataLoader(colored_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
colored_test_set = ColoredDataset(test_set, classes=10, colors=colored_train_set.colors, std=args.color_std)
# test_loader = DataLoader(colored_test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(colored_test_set, batch_size=len(colored_test_set), shuffle=False, num_workers=4, pin_memory=True)

# measure bias
color_fn = lambda x: x.view(x.size(0), x.size(1), -1).max(2)[0]  # color of digit
color_dim = 3  # rgb
bias = measure_bias(train_loader, test_loader, color_fn, color_dim, opt)[0]
print('Color bias of Colored MNIST = {:.3f}'.format(bias + 0))

# # measure generalization
# model = create_mnist_model(args.model)
# acc, gt_acc = measure_generalization(train_loader, [test_loader, gt_test_loader], model, opt)
# print('Test accuracy on Colored MNIST = {:.2%}'.format(acc))
# print('Generalization on Grayscale MNIST = {:.2%}'.format(gt_acc))

# save feature, label, color
color_fn = lambda x: x.view(x.size(0), x.size(1), -1).max(2)[0]  # color of digit
features, labels, colors = get_feature(train_loader, test_loader, color_fn, opt)
save_file = {'feature': features,
             'label': labels,
             'color': colors}
save_path = './feature'
create_folder(save_path)
save_pkl(save_file, os.path.join(save_path, 'feature_{}.pkl'.format(args.color_std)))