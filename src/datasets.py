"""PyTorch dataset and transforms for bin images."""
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .config import IMAGENET_MEAN, IMAGENET_STD


class BinDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.fnames = df["fname"].tolist()
        self.labels = df["quantity"].astype(int).tolist()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        img = Image.open(self.img_dir / self.fnames[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def build_transforms(size, train):
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return T.Compose([T.Resize((size, size)), *norm])
    # Mild augmentation only: aggressive crops could cut an item out and change the label.
    return T.Compose([
        T.Resize((size, size)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05))], p=0.5),
        T.ColorJitter(0.2, 0.2, 0.15, 0.02),
        *norm,
    ])


def make_loaders(paths, size, batch_size, num_workers, max_train=None, seed=42):
    df = pd.read_csv(paths["splits"])
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    if max_train and len(tr) > max_train:
        tr = tr.sample(n=max_train, random_state=seed)
    pin = torch.cuda.is_available()
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin,
              persistent_workers=num_workers > 0)
    loaders = {
        "train": DataLoader(BinDataset(tr, paths["images"], build_transforms(size, True)), shuffle=True,
                            drop_last=len(tr) > batch_size, **kw),
        "val": DataLoader(BinDataset(va, paths["images"], build_transforms(size, False)), shuffle=False, **kw),
        "test": DataLoader(BinDataset(te, paths["images"], build_transforms(size, False)), shuffle=False, **kw),
    }
    return loaders, tr
