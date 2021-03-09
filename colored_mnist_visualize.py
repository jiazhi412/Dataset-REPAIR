# vary training color std, keep same testing color std
test_color_std = 0.5
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

from utils.datasets import ColoredDataset
from utils.measure import *
from utils.models import *
from utils.save import *

from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread

def showImagesHorizontally(list_of_files):
    fig = figure()
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a=fig.add_subplot(1,number_of_files,i+1)
        image = imread(list_of_files[i])
        imshow(image,cmap='Greys_r')
        axis('off')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--color-std', type=float, default=1e-1)
parser.add_argument('--batch-size', default=128, type=int)
args = parser.parse_args()

opt = vars(parser.parse_args())
opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')

root = './data'
if not os.path.exists(root):
    os.mkdir(root)

# load data
test_set = datasets.MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())

# biased datasets, i.e. colored mnist
print('Coloring MNIST dataset with standard deviation = {:.2f}'.format(args.color_std))
colored_test_set = ColoredDataset(test_set, classes=10, colors=colored_train_set.colors, std=test_color_std)

image1 = colored_test_set[0]
image2 = colored_test_set[1]

list = [image1, image2]

# visualization
row = 2
col = 30
show_per_digits = 3



showImagesHorizontally(list)




