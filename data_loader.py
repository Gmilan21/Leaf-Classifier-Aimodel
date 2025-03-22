import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data(data_dir='dataset', batch_size=32, augment=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if augment:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  
            transforms.CenterCrop(128),  
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  
            transforms.RandomResizedCrop(128, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            normalize,
        ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
