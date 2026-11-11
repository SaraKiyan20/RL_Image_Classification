
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# 1. Select the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# 2. Load pretrained ResNet18
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

print("Pretrained ResNet18 loaded!")

# 3. Freeze all existing layers
for param in model.parameters():
    param.requires_grad = False

# 4. Replace the final classification layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

# 5. Move model to GPU
model = model.to(device)

# 6. Test the model with a sample input
sample = torch.randn(2, 3, 224, 224).to(device)

with torch.no_grad():
    output = model(sample)

print("Output shape:", output.shape)
print("Model device:", next(model.parameters()).device)
print("ResNet18 is ready!")
