import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset


class ForceDataset(Dataset):

    def __init__(self, image_dir, stress_dir, force_file, image_size=224):

        self.image_dir = image_dir
        self.stress_dir = stress_dir
        self.image_size = image_size

        self.data = pd.read_csv(force_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        image_path = os.path.join(self.image_dir, row["image"])
        stress_path = os.path.join(self.stress_dir, row["stress"])

        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        stress = cv2.imread(stress_path, 0)
        stress = cv2.resize(stress, (self.image_size, self.image_size))

        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        stress = torch.tensor(stress).unsqueeze(0).float() / 255.0

        force = torch.tensor([row["force_x"], row["force_y"]]).float()

        return image, stress, force
