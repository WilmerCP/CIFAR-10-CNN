# datos.py
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CLASES = ["avión", "auto", "pájaro", "gato", "ciervo",
          "perro", "rana", "caballo", "barco", "camión"]

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def get_loaders(batch_size=64, val_size=5000, seed=42):
    # Dos copias del mismo dataset: una con augmentation (train) y otra sin (val)
    train_full = datasets.CIFAR10("data", train=True, download=True, transform=train_tf)
    val_full = datasets.CIFAR10("data", train=True, download=True, transform=eval_tf)
    test_ds = datasets.CIFAR10("data", train=False, download=True, transform=eval_tf)

    # Mezcla fija de índices para que la división sea siempre la misma
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(train_full), generator=g).tolist()
    train_ds = Subset(train_full, idx[val_size:])
    val_ds = Subset(val_full, idx[:val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    tr, va, te = get_loaders()
    print(len(tr.dataset), len(va.dataset), len(te.dataset))
    x, y = next(iter(tr))
    print(x.shape, y.shape)