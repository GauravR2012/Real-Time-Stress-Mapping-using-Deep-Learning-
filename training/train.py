# training/train.py

import torch
from torch.utils.data import DataLoader

from models.fre_network import FREModel
from dataset.dataset_loader import ForceDataset
from training.loss import compute_loss

dataset = ForceDataset("data/images","data/stress","data/forces.csv")

loader = DataLoader(dataset,batch_size=16,shuffle=True)

model = FREModel()

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

for epoch in range(10):

    for img,stress,force in loader:

        pred_stress,pred_force = model(img)

        loss = compute_loss(pred_stress,stress,pred_force,force)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch",epoch,"loss",loss.item())
