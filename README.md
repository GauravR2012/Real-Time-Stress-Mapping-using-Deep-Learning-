# Real-Time Stress Mapping Using Deep Learning

Deep learning framework for predicting stress distribution and contact force from deformation images of a soft robotic Fin Ray gripper.

---

## Overview

Soft robotic grippers exhibit complex deformation behavior that is difficult to model analytically. This project implements a deep learning architecture capable of predicting contact force and stress distribution directly from visual deformation images.

The model follows the architecture proposed in:

De Barrie et al., "A Deep Learning Method for Vision Based Force Prediction of a Soft Fin Ray Gripper Using Simulation Data", Frontiers in Robotics and AI (2021).

The system learns a representation of gripper deformation using a convolutional encoder and decodes this representation into:

• Stress distribution maps  
• Contact force estimates

---

## Architecture

The model consists of three main components:

1. Deformation Encoder (CNN)
2. Stress Decoder (U-Net style)
3. Force Decoder (Fully Connected Network)

Pipeline:

Deformation Image  
→ Encoder  
→ Latent Representation  

Branch 1: Stress Decoder → Stress Map  
Branch 2: Force Decoder → Contact Force

---

## Dataset

The training data consists of deformation images generated from Finite Element Analysis (FEA) simulations of a Fin Ray gripper interacting with objects of varying geometry.

Each sample contains:

• Deformation image  
• Ground truth stress map  
• Contact force value

Dataset format:
data/
images/
stress_maps/
forces.csv


The dataset used in the original experiments is not included in this repository.

---

## Training

Training uses a joint loss function combining stress reconstruction and force prediction:

L = αs * stress_loss + αf * force_loss

where

αs = 1  
αf = 10

Optimization:

Optimizer: Adam  
Learning Rate: 1e-4  
Batch Size: 16  
Epochs: 10

---

## Inference

Run inference on a new deformation image:
python inference/predict.py --image path/to/image.png


Outputs:

• predicted stress map  
• predicted contact force

---

## Repository Structure
models/
encoder, decoder and full network

dataset/
dataset loader

training/
training scripts

inference/
inference pipeline

utils/
visualization tools

---

## Applications

• Soft robotic grasp analysis  
• Real-time force sensing without physical sensors  
• Robotic manipulation of delicate objects  
• Simulation-driven robotics learning

---

## License

MIT License
