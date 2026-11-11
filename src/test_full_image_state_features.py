
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from torchvision.datasets import STL10


# -----------------------------
# 1. Settings
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------
# 2. Load full-image model
# -----------------------------

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    10
)

model.load_state_dict(
    torch.load(
        "models/resnet18_stl10_best.pth",
        map_location=device,
        weights_only=True
    )
)

# Remove classification layer
model.fc = nn.Identity()

model = model.to(device)
model.eval()


# -----------------------------
# 3. Load one test image
# -----------------------------

dataset = STL10(
    root="data",
    split="test",
    download=False,
    transform=transforms.ToTensor()
)

image, label = dataset[0]

print("\nOriginal image shape:", image.shape)
print("Class:", dataset.classes[label])


# -----------------------------
# 4. Create low-resolution overview
# -----------------------------

image = image.unsqueeze(0).to(device)

overview = torch.nn.functional.interpolate(
    image,
    size=(32, 32),
    mode="bilinear",
    align_corners=False
)

overview = torch.nn.functional.interpolate(
    overview,
    size=(224, 224),
    mode="bilinear",
    align_corners=False
)


# ImageNet normalization
mean = torch.tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1).to(device)

std = torch.tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1).to(device)

overview = (overview - mean) / std


# -----------------------------
# 5. Extract features
# -----------------------------

with torch.no_grad():

    features = model(overview)


# -----------------------------
# 6. Display results
# -----------------------------

print("\nFeature Extraction Results")
print("---------------------------")

print("Input shape:", overview.shape)
print("Output shape:", features.shape)
print("Number of features:", features.shape[1])

print("\nFull-image state feature extractor is working!")
