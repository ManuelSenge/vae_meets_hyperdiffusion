from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import torch


class MNISTDataset(Dataset):
    def __init__(self, train, shuffle=False):
        if train:
            data = pd.read_csv('./data/mnist_train.csv')
        else:
            data = pd.read_csv('./data/mnist_test.csv')

        if shuffle:
            data = data.sample(frac = 1)

        self.labels = data['label'].values.reshape((-1, 1))
        images = data.drop(columns=['label']).values
        self.images = images.reshape((-1, 28, 28)).astype('float32')/255

    def __len__(self):
        return len(self.labels)
       
    def __getitem__(self, index):
        return torch.Tensor([self.labels[index]]), torch.Tensor([self.images[index]])

    def plot_image_by_idx(self, idx):
        plt.imshow(self.images[idx], cmap='gray')
        plt.title(f'label: {self.labels[idx]}')
    
    def plot_image(image):
        plt.imshow(image, cmap='gray')

       

    

    