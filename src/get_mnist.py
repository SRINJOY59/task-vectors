import os
import torch
from torchvision import transforms
from datasets.mnist import MNIST

def download_and_save_mnist(location='data'):
    os.makedirs(location, exist_ok=True)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_data = MNIST(preprocess=preprocess, location=location)

    print("MNIST dataset has been downloaded and stored in:", location)

download_and_save_mnist()