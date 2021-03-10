# vary training color std, keep same testing color std
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
import numpy as np
import matplotlib.pyplot as plt

def showImagesHorizontally(list_of_files):
    fig = figure()
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a=fig.add_subplot(1,number_of_files,i+1)
        image = list_of_files[i]
        image = np.moveaxis(image, 0,2)
        imshow(image,cmap='Greys_r')
        axis('off')

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(1-image)
        # a.set_title(title)
        plt.axis('off')
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

std = 0.1
two_color = True

root = './data'
if not os.path.exists(root):
    os.mkdir(root)

# load data
test_set = datasets.MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())

# biased datasets, i.e. colored mnist
print('Coloring MNIST dataset with standard deviation = {:.2f}'.format(std))
colored_test_set = ColoredDataset(test_set, classes=10, colors=[0, 1], std=std, two_color=two_color)

list = []
pointer = 0
flag = 0
for i in range(len(test_set)):
    image, label = colored_test_set[i]
    if label == pointer:
        image = np.moveaxis(image.numpy(), 0, 2)
        list.append(image)
        flag += 1
        if flag == 2:
            pointer += 1
            flag = 0

# visualization
row = 2
col = 30
show_per_digits = 3



show_images(list)




