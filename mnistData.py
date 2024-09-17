import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
#
# # Define a transformation to convert the images to tensors and normalize them
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))  # Mean and std deviation for normalization
# ])
#
# # Load the training and test datasets
# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#
# # Create data loaders for batching
# train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
#
# # Check the shape of data
# images, labels = next(iter(train_loader))
# print(images.shape)  # [64, 1, 28, 28] -> [Batch Size, Channels, Height, Width]
# print(labels.shape)  # [64]
#
#
# # for batch_idx, (data, target) in enumerate(train_loader):
# #     print(f'Batch: {batch_idx}, Data shape: {data.shape}, Target shape: {target.shape}')
# #     # Your training loop here
#





import torch
from torchvision import datasets, transforms
import torch.nn.functional as F

# Define a transformation to apply to the images (optional)
transform = transforms.Compose([
    transforms.ToTensor(),          # Convert the image to a tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images
])

# Download the training and test datasets
train_dataset = datasets.MNIST(root='data', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='data', train=False, download=False, transform=transform)



if __name__ == "__main__":
    # Print out some information about the datasets
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    # Example of accessing a sample
    sample, label = train_dataset[1]
    print(f"Sample image shape: {sample.shape}")
    print(f"Label: {label}")






