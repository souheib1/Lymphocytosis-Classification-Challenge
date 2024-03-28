import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms


class SimCLR(nn.Module):
    def __init__(self, encoder, head, nb_channels, nb_rows, nb_columns):
      super().__init__()

      self.encoder = encoder
      self.head = head

      self.nb_channels = nb_channels
      self.nb_rows = nb_rows
      self.nb_columns = nb_columns

      self.transform = transforms.Compose([
          transforms.RandomCrop(int(nb_rows * 0.9)),
          transforms.Resize((nb_rows, nb_columns)),
          transforms.RandomHorizontalFlip(p=0.5),
          transforms.RandomVerticalFlip(p=0.5),
          transforms.RandomApply([transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)], p=0.5),
          transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5)
          ])

    def augment(self, x):
      return self.transform(x)

    def forward(self, x):
        view = self.augment(x)
        h = self.encoder(view)
        z = self.head(h)
        return z