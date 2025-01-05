import os

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class BridgeDataset(Dataset):
    def __init__(self, image_dir, label_path):
        self.image_dir = image_dir
        with open(label_path, 'r') as f:
            lines = f.readlines()
        labels = []
        for line in lines:
            image_name, class_id, _ = line.strip().split()
            labels.append((image_name, int(class_id)))
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_name, class_id = self.labels[index]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path)
        image = image.resize((128, 256))
        image = torch.from_numpy(np.array(image) / 255.0).float()
        image = image.permute(2, 0, 1)
        label = torch.tensor(class_id, dtype=torch.long)

        return image, label
