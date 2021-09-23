from torch.utils.data import Dataset
from torchvision import datasets, models, transforms


def get_augmentation(stage):
    if stage == 'train':
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomRotation(30,),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return transform

class CustomTransform(Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transform = transforms

    def __getitem__(self, idx):
        sample, target = self.dataset[idx]
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.dataset)